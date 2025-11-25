import numpy as np
from scipy.stats import norm
from typing import Union, Dict, Any, List
import json
import pandas as pd

from langchain.tools import tool

from .tool_registry import register_tool
from .utils import load_json_as_df


def _bsm_calculator(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the Black-Scholes price for European call or put options.

    Args:
        option_type (str): Type of the option, either 'call' or 'put'
        S (float): Current price of the underlying asset
        K (float): Strike price of the option
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying asset (annualized standard deviation)

    Returns:
        str: The calculated option price as a string
    """
    option_type = option_type.lower()

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        option_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return option_price

@register_tool(tags=["bsm","price","batch"], roles=["calculator"])
@tool("batch_bsm_calculator")
def batch_bsm_calculator(csv_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> str:
    """
    Batch calculates Black-Scholes option prices for multiple options from CSV data.

    Args:
        csv_data: JSON string, dict, or list of dicts with columns: option_type, S, K, T, r, sigma

    Returns:
        str: JSON string with state_update containing bsm_results
    """
    try:
        df = load_json_as_df(csv_data)
        if df is False:
            return json.dumps({"errors": [f"csv_data must be a string, dict, or list, got {type(csv_data)}"]})

        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"errors": [f"Missing required columns: {missing_cols}"]}

        def calc_row(row):
            return _bsm_calculator(
                row['option_type'], 
                row['S'], 
                row['K'], 
                row['T'], 
                row['r'], 
                row['sigma']
            )
        df['BSM_Price'] = df.apply(
            calc_row, axis=1
        )
        result = {"bsm_results": df.to_json(orient='records', date_format='iso')}
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"errors": [f"batch_bsm_calculator error: {e}"]})


# @register_tool(tags=["greeks"], roles=["calculator"])
# @tool("greeks_calculator")
def _greeks_calculator(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
    """
    Calculates option Greeks for a European call or put using the Black-Scholes model.

    Args:
        option_type (str): 'call' or 'put'
        S (float): Spot price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free rate (annualized)
        sigma (float): Volatility (annualized)

    Returns:
        str: JSON string containing price, delta, gamma, vega, rho, theta
    """
    try:
        option_type = option_type.lower()

        if T <= 0 or sigma <= 0:
            raise ValueError("T and sigma must be positive")

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = float(norm.cdf(d1))
            gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
            vega = float(S * norm.pdf(d1) * np.sqrt(T))
            rho = float(K * T * np.exp(-r * T) * norm.cdf(d2))
            theta = float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)))
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = float(norm.cdf(d1) - 1)
            gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
            vega = float(S * norm.pdf(d1) * np.sqrt(T))
            rho = float(-K * T * np.exp(-r * T) * norm.cdf(-d2))
            theta = float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)))
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        results = {
            "price": float(price),
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho": rho,
            "theta": theta,
        }

        return results
    except Exception as e:
        raise ValueError(str(e))

@register_tool(tags=["greeks","batch"], roles=["calculator"])
@tool("batch_greeks_calculator")
def batch_greeks_calculator(csv_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> str:
    """
    Batch calculates greeks for multiple options from CSV data.

    Args:
        csv_data: JSON string, dict, or list of dicts with columns: option_type, S, K, T, r, sigma

    Returns:
        str: JSON string with state_update containing greeks_results
    """
    try:
        df = load_json_as_df(csv_data)
        if df is False:
            return json.dumps({"errors": [f"csv_data must be a string, dict, or list, got {type(csv_data)}"]})
        
        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return json.dumps({"errors": [f"Missing required columns: {missing_cols}"]})
        
        def calc_row(row):
            res = _greeks_calculator(
                row['option_type'], 
                row['S'], 
                row['K'], 
                row['T'], 
                row['r'], 
                row['sigma'], 
            )
            return res
        
        expanded = df.apply(calc_row, axis=1).apply(pd.Series)
        result_cols = ['price','delta','gamma','vega','rho','theta']
        for col in result_cols:
            if col not in expanded:
                expanded[col] = pd.NA
        df = pd.concat([df, expanded[result_cols]], axis=1)
        result = {"greeks_results": df.to_json(orient='records', date_format='iso')}
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({"errors": [f"batch_greeks_calculator error: {e}"]})


@register_tool(tags=["sensitivity"], roles=["calculator"])
@tool("sensitivity_test")
def sensitivity_test(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
    """
    Runs a sensitivity test by perturbing the spot price around S and computing Greeks at each perturbed spot.

    Tests spot price changes from -2.5% to +2.5% in 0.5% increments (11 points total).

    Args:
        option_type (str): 'call' or 'put'
        S (float): Base spot price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free rate
        sigma (float): Volatility

    Returns:
        str: JSON list of sensitivity results, each entry contains spot_change and greeks
    """
    try:
        spot_changes = np.arange(-0.025, 0.026, 0.005)
        results = []

        for change in spot_changes:
            new_S = S * (1 + float(change))
            # Call greeks_calculator directly (as a function, not via tool API)
            greeks_json = greeks_calculator.invoke(
                {
                    "option_type": option_type, 
                    "S": new_S, 
                    "K": K, 
                    "T": T, 
                    "r": r, 
                    "sigma": sigma,
                }
            )
            greeks = json.loads(greeks_json)

            entry = {"spot_change": float(change)}
            if 'error' in greeks:
                entry['error'] = greeks['error']
            else:
                for k in ("price", "delta", "gamma", "vega", "rho", "theta"):
                    entry[k] = greeks.get(k)

            results.append(entry)

        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})
