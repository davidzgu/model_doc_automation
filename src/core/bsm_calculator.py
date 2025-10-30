# -*- coding: utf-8 -*-
"""
Black-Scholes-Merton Calculator

All BSM calculation tools for option pricing:
- bsm_calculator: Calculate single option price
- batch_bsm_calculator: Calculate prices for multiple options
- greeks_calculator: Calculate option Greeks (delta, gamma, vega, rho, theta)
- sensitivity_test: Run spot price sensitivity analysis
"""
import numpy as np
from scipy.stats import norm
from langchain.tools import tool
import pandas as pd
import json
from typing import Union, Dict, Any


@tool
def bsm_calculator(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
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

    return str(option_price)


@tool
def batch_bsm_calculator(csv_data: Union[str, Dict[str, Any]]) -> Dict[str,str]:
    """
    Batch calculates Black-Scholes option prices for multiple options from CSV data.

    Args:
        csv_data: JSON string or dict of CSV data with columns: option_type, S, K, T, r, sigma

    Returns:
        str: Markdown table with input parameters and calculated option prices for each row
    """
    try:
        if isinstance(csv_data, str):
            data = json.loads(csv_data)
        elif isinstance(csv_data, dict):
            data = csv_data
        else:
            return f"Error: csv_data must be a string or dict, got {type(csv_data)}"

        df = pd.DataFrame(data)

        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            result = {
                "type": "Error Information",
                "data": "Error: Missing required columns: {missing_cols}"
            }
            return result

        def calc_row(row):
            return bsm_calculator.func(
                option_type=row['option_type'], 
                S=row['S'], 
                K=row['K'], 
                T=row['T'], 
                r=row['r'], 
                sigma=row['sigma'], 
            )
        
        df['BSM_Price'] = df.apply(
            calc_row, axis=1
        )


        result = {
            "type": "Black-Scholes Option Pricing Results",
            "data": df.to_json(orient="records")
        }

        return result

    except json.JSONDecodeError as e:
        result = {
            "type": "Error Information",
            "data": f"Error: Invalid JSON format. {str(e)}"
        }
        return result
    except Exception as e:
        result = {
            "type": "Error Information",
            "data": f"Error calculating option prices: {str(e)}"
        }
        return result



# @tool
# def batch_bsm_calculator(csv_data: Union[str, Dict[str, Any]]) -> str:
#     """
#     Batch calculates Black-Scholes option prices for multiple options from CSV data.

#     Args:
#         csv_data: JSON string or dict of CSV data with columns: option_type, S, K, T, r, sigma

#     Returns:
#         str: Markdown table with input parameters and calculated option prices for each row
#     """
#     try:
#         if isinstance(csv_data, str):
#             data = json.loads(csv_data)
#         elif isinstance(csv_data, dict):
#             data = csv_data
#         else:
#             return f"Error: csv_data must be a string or dict, got {type(csv_data)}"

#         df = pd.DataFrame(data)

#         required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             return f"Error: Missing required columns: {missing_cols}"

#         prices = []
#         for idx, row in df.iterrows():
#             option_type = str(row['option_type']).lower()
#             S = float(row['S'])
#             K = float(row['K'])
#             T = float(row['T'])
#             r = float(row['r'])
#             sigma = float(row['sigma'])

#             d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#             d2 = d1 - sigma * np.sqrt(T)

#             if option_type == 'call':
#                 price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#             elif option_type == 'put':
#                 price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
#             else:
#                 price = None

#             prices.append(price)

#         df['BSM_Price'] = prices

#         result = "## Black-Scholes Option Pricing Results\n\n"
#         result += df.to_markdown(index=False)

#         return result

#     except json.JSONDecodeError as e:
#         return f"Error: Invalid JSON format. {str(e)}"
#     except Exception as e:
#         return f"Error calculating option prices: {str(e)}"


@tool
def greeks_calculator(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
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
            return json.dumps({"error": "T and sigma must be positive"})

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
            return json.dumps({"error": "option_type must be 'call' or 'put'"})

        results = {
            "price": float(price),
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho": rho,
            "theta": theta,
        }

        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
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
            greeks_json = greeks_calculator(option_type, new_S, K, T, r, sigma)
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
