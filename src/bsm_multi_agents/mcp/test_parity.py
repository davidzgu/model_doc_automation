import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from .data_exporter import save_to_local
from .option_pricer import calc_bsm_price

def verify_put_call_parity(input_path: str) -> str:
    """
    Identifies actual Call-Put pairs in the input data and verifies the Put-Call Parity relationship.
    Formula: C + K * exp(-r * T) = P + S
    
    Args:
        input_path (str): Absolute path to the input CSV file. 
            Required columns: underlying, S, K, T, r, sigma, option_type, price (or BSM_price)
    """
    df = pd.read_csv(input_path)
    
    
    # Fallback: if no price at all, calculate it so the tool still works
    if 'price' not in df.columns:
        df['price'] = calc_bsm_price(df['S'], df['K'], df['T'], df['r'], df['sigma'], df['option_type'])

    # Group by underlying features to find pairs
    match_cols = ['underlying', 'S', 'K', 'T', 'r', 'sigma']
    
    # Separate calls and puts
    calls = df[df['option_type'].str.lower() == 'call'].copy()
    puts = df[df['option_type'].str.lower() == 'put'].copy()
    
    # Merge on matching columns to find pairs
    pairs = pd.merge(
        calls, 
        puts, 
        on=match_cols, 
        suffixes=('_call', '_put')
    )
    
    if pairs.empty:
        pairs['lhs'] = None
        pairs['rhs'] = None
        pairs['abs_diff'] = None
        pairs['is_parity_valid'] = None
        pairs['arbitrage_opportunity'] = None
    else:
        # 2. Calculate Left Hand Side (LHS) and Right Hand Side (RHS) using the matched prices
        # LHS = C + K * exp(-r * T)
        # RHS = P + S
        C = pairs['price_call']
        P = pairs['price_put']
        K = pairs['K']
        S = pairs['S']
        r = pairs['r']
        T = pairs['T']

        pairs['lhs'] = C + K * np.exp(-r * T)
        pairs['rhs'] = P + S
        
        # 3. Calculate Absolute and Relative Discrepancy
        pairs['abs_diff'] = np.abs(pairs['lhs'] - pairs['rhs'])
        pairs['is_parity_valid'] = pairs['abs_diff'] < 1e-4
        
        # 4. Identify Potential Arbitrage Side
        pairs['arbitrage_opportunity'] = np.where(
            pairs['lhs'] > pairs['rhs'] + 0.01, "Short Call / Long Put",
            np.where(pairs['rhs'] > pairs['lhs'] + 0.01, "Long Call / Short Put", "None")
        )
    
    result_df = pairs

    output_path = save_to_local(
        result_df, 
        folder_name="analytics", 
        prefix="put_call_parity"
    )
    
    return output_path