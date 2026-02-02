import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, Tuple

from .data_exporter import save_to_local


def _calc_d1_d2(
    S: pd.Series, 
    K: pd.Series, 
    T: pd.Series, 
    r: pd.Series, 
    sigma: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    # Internal math only
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def calc_delta(
    S: pd.Series, 
    K: pd.Series, 
    T: pd.Series, 
    r: pd.Series, 
    sigma: pd.Series, 
    option_type: pd.Series
) -> pd.Series:
    # Calculate the Delta column for the given options.
    d1, _ = _calc_d1_d2(S, K, T, r, sigma)
    return np.where(option_type.str.lower() == "call", norm.cdf(d1), norm.cdf(d1) - 1)


def calc_gamma(
    S: pd.Series, 
    K: pd.Series, 
    T: pd.Series, 
    r: pd.Series, 
    sigma: pd.Series
) -> pd.Series:
    # Calculate the Gamma column (same formula for calls and puts).
    d1, _ = _calc_d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def calc_vega(
    S: pd.Series, 
    K: pd.Series, 
    T: pd.Series, 
    r: pd.Series, 
    sigma: pd.Series
) -> pd.Series:
    # Calculate the Vega column (same formula for calls and puts).
    d1, _ = _calc_d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)/100.0

def calc_theta(
    S: pd.Series, 
    K: pd.Series, 
    T: pd.Series, 
    r: pd.Series, 
    sigma: pd.Series, 
    option_type: pd.Series
) -> pd.Series:
    # Calculate the Theta column for the given options.
    d1, d2 = _calc_d1_d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    call_theta = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)/365.0
    put_theta = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)/365.0
    return np.where(option_type.str.lower() == "call", call_theta, put_theta)

def calc_rho(
    S: pd.Series, 
    K: pd.Series, 
    T: pd.Series, 
    r: pd.Series, 
    sigma: pd.Series, 
    option_type: pd.Series
) -> pd.Series:
    # Calculate the Rho column for the given options.
    _, d2 = _calc_d1_d2(S, K, T, r, sigma)
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return np.where(option_type.str.lower() == "call", call_rho, put_rho)

def calc_bsm_price(
    S: pd.Series, 
    K: pd.Series, 
    T: pd.Series, 
    r: pd.Series, 
    sigma: pd.Series, 
    option_type: pd.Series
) -> pd.Series:
    # Calculate the price column for the given options.
    d1, d2 = _calc_d1_d2(S, K, T, r, sigma)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(option_type.str.lower() == "call", call_price, put_price)


def calculate_option_analytics(input_path: str) -> str:
    """
    Perform batch Black-Scholes pricing and Greeks calculation on a CSV file.

    This tool reads a CSV, computes theoretical prices and the five major Greeks 
    (Delta, Gamma, Vega, Theta, Rho), saves the results to a new local file, 
    and returns the absolute path.

    Args:
        input_path (str): Absolute path to the input CSV file. 
            Required columns:
            - 'S': Underlying asset price (e.g., 100.50)
            - 'K': Strike price of the option (e.g., 100.00)
            - 'T': Time to maturity in years (e.g., 0.25 for 3 months)
            - 'r': Annualized risk-free interest rate (e.g., 0.05 for 5%)
            - 'sigma': Annualized volatility (e.g., 0.20 for 20%)
            - 'option_type': Type of the option, either 'call' or 'put' (case-insensitive)

    Returns:
        str: The absolute local file path of the processed CSV containing original data 
             plus columns: ['price', 'delta', 'gamma', 'vega', 'theta', 'rho'].
    """
    # Defensive copy to keep the original input state immutable
    df = pd.read_csv(input_path)
    res = df.copy()
    
    # Mapping dataframe columns to local variables for readability
    S, K, T, r, v = res['S'], res['K'], res['T'], res['r'], res['sigma']
    opt_type = res['option_type']

    res['price'] = calc_bsm_price(S, K, T, r, v, opt_type)

    # 2. Greeks Calculation using specialized functions
    res['delta'] = calc_delta(S, K, T, r, v, opt_type)
    res['gamma'] = calc_gamma(S, K, T, r, v)
    res['vega'] = calc_vega(S, K, T, r, v)
    res['theta'] = calc_theta(S, K, T, r, v, opt_type)
    res['rho'] = calc_rho(S, K, T, r, v, opt_type)

    return save_to_local(res, folder_name="analytics", prefix="analyzed_options")


