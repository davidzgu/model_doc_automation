import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from .data_exporter import save_to_local
from .option_pricer import calc_bsm_price

def verify_put_call_parity(input_path: str) -> str:
    """
    Test the Put-Call Parity relationship for a given set of options.
    Formula: C + K * exp(-r * T) = P + S
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
             plus columns: ['lhs', 'rhs', 'abs_diff', 'is_parity_valid', 'arbitrage_opportunity'].
    """
    # 1. Prepare data
    df = pd.read_csv(input_path)
    S = df['S']
    K = df['K']
    T = df['T']
    r = df['r']
    v = df['sigma']
    
    # We need both call and put prices for the same S, K, T
    # Assuming the input df has 'call_price' and 'put_price' columns

    call_price = calc_bsm_price(S, K, T, r, v, pd.Series('call', index=df.index))
    put_price = calc_bsm_price(S, K, T, r, v, pd.Series('put', index=df.index))
    C = call_price
    P = put_price

    # 2. Calculate Left Hand Side (LHS) and Right Hand Side (RHS)
    df['lhs'] = C + K * np.exp(-r * T)
    df['rhs'] = P + S
    
    # 3. Calculate Absolute and Relative Discrepancy
    df['abs_diff'] = np.abs(df['lhs'] - df['rhs'])
    df['is_parity_valid'] = df['abs_diff'] < 1e-4 # Tolerance for float precision
    
    # 4. Identify Potential Arbitrage Side
    # If LHS > RHS: Call is overpriced relative to Put (Short Call, Long Put, Long Stock)
    # If RHS > LHS: Put is overpriced relative to Call
    df['arbitrage_opportunity'] = np.where(
        df['lhs'] > df['rhs'] + 0.01, "Short Call / Long Put",
        np.where(df['rhs'] > df['lhs'] + 0.01, "Long Call / Short Put", "None")
    )
    
    return save_to_local(df, folder_name="analytics", prefix="put_call_parity")