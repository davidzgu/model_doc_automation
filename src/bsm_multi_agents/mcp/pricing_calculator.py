import pandas as pd
import os
import numpy as np
from scipy.stats import norm

def bsm_calculator(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the Black-Scholes price for European call or put options.
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

def calculate_bsm_to_file(input_path: str, output_dir: str) -> str:
    """
    Reads the CSV at input_path, calculates BSM prices, and saves the result to output_dir.
    Returns the path to the result file.
    """
    try:
        if not os.path.exists(input_path):
            return f"Error: Input file not found at {input_path}"

        df = pd.read_csv(input_path)
        
        # Validate required columns
        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Error: Missing required columns: {missing_cols}"

        # Calculation Logic
        def calc_row(row):
            return bsm_calculator(
                row['option_type'], 
                row['S'], 
                row['K'], 
                row['T'], 
                row['r'], 
                row['sigma']
            )
        
        df['BSM_price'] = df.apply(calc_row, axis=1)
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", "_bsm_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        df.to_csv(output_path, index=False)
        
        return os.path.abspath(output_path)
        
    except Exception as e:
        return f"Error processing file: {str(e)}"


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
        str: JSON string containing BSM_price, delta, gamma, vega, rho, theta
    """

    def _error_result(message: str) -> dict:
        return {
            "BSM_price": np.nan,
            "delta": np.nan,
            "gamma": np.nan,
            "vega": np.nan,
            "rho": np.nan,
            "theta": np.nan,
            "error": message,
        }


    try:
        option_type = option_type.lower()

        if T <= 0 or sigma <= 0:
            return _error_result("T and sigma must be positive")
        if S <= 0 or K <= 0:
            return _error_result("S and K must be positive")
        if option_type not in ("call", "put"):
            return _error_result("option_type must be 'call' or 'put'")
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            BSM_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = float(norm.cdf(d1))
            gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
            vega = float(S * norm.pdf(d1) * np.sqrt(T))
            rho = float(K * T * np.exp(-r * T) * norm.cdf(d2))
            theta = float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)))
        else: # put
            BSM_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = float(norm.cdf(d1) - 1)
            gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
            vega = float(S * norm.pdf(d1) * np.sqrt(T))
            rho = float(-K * T * np.exp(-r * T) * norm.cdf(-d2))
            theta = float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)))

        results = {
            "BSM_price": float(BSM_price),
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho": rho,
            "theta": theta,
            "error": None,
        }

        return results
    except Exception as e:
        return _error_result(str(e))

def calculate_greeks_to_file(input_path: str, output_dir: str = "./output") -> str:
    """
    Reads the CSV at input_path, calculates greeks, and saves the result to output_dir.
    Returns the path to the result file.
    """
    try:
        if not os.path.exists(input_path):
            return f"Error: Input file not found at {input_path}"

        df = pd.read_csv(input_path)

        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Error: Missing required columns: {missing_cols}"
        
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
        result_cols = ['BSM_price','delta','gamma','vega','rho','theta','error']
        for col in result_cols:
            if col not in expanded:
                expanded[col] = pd.NA
        df = pd.concat([df, expanded[result_cols]], axis=1)

        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", "_greeks_results.csv")
        output_path = os.path.join(output_dir, filename)

        df.to_csv(output_path, index=False)

        return os.path.abspath(output_path)

        
    except Exception as e:
        return f"Error processing file: {str(e)}"
