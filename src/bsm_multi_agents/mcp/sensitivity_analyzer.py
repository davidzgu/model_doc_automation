import pandas as pd
import numpy as np
import os
from typing import List, Dict
from scipy.stats import norm

from .option_pricer import calc_bsm_price
from .data_exporter import save_to_local


TEST_SCENARIOS = {
    "name": "spot_bump_-0.05",
    "shocks": [
        {"parameter": "S", "type": "relative", "value": -0.05},
    ]
}

DEFAULT_SENSITIVITY_SCENARIOS = [
    {
        "name": "spot_bump_-0.05",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": -0.05},
        ]
    },
    {
        "name": "spot_bump_-0.02",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": -0.02},
        ]
    },
    {
        "name": "spot_bump_-0.01",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": -0.01},
        ]
    },
    {
        "name": "spot_bump_0",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": 0},
        ]
    },
    {
        "name": "spot_bump_+0.01",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": +0.01},
        ]
    },
    {
        "name": "spot_bump_+0.02",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": +0.02},
        ]
    },
    {
        "name": "spot_bump_+0.05",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": +0.05},
        ]
    },
    {
        "name": "vol_bump_-0.05",
        "shocks": [
            {"parameter": "sigma", "type": "absolute", "value": -0.05},
        ]
    },
    {
        "name": "vol_bump_-0.02",
        "shocks": [
            {"parameter": "sigma", "type": "absolute", "value": -0.02},
        ]
    },
    {
        "name": "vol_bump_-0.01",
        "shocks": [
            {"parameter": "sigma", "type": "absolute", "value": -0.01},
        ]
    },
    {
        "name": "vol_bump_0",
        "shocks": [
            {"parameter": "sigma", "type": "absolute", "value": 0},
        ]
    },
    {
        "name": "vol_bump_+0.01",
        "shocks": [
            {"parameter": "sigma", "type": "absolute", "value": +0.01},
        ]
    },
    {
        "name": "vol_bump_+0.02",
        "shocks": [
            {"parameter": "sigma", "type": "absolute", "value": +0.02},
        ]
    },
    {
        "name": "vol_bump_+0.05",
        "shocks": [
            {"parameter": "sigma", "type": "absolute", "value": +0.05},
        ]
    },
    {
        "name": "rate_bump_-0.05",
        "shocks": [
            {"parameter": "r", "type": "absolute", "value": -0.05},
        ]
    },
    {
        "name": "rate_bump_-0.02",
        "shocks": [
            {"parameter": "r", "type": "absolute", "value": -0.02},
        ]
    },
    {
        "name": "rate_bump_-0.01",
        "shocks": [
            {"parameter": "r", "type": "absolute", "value": -0.01},
        ]
    },
    {
        "name": "rate_bump_0",
        "shocks": [
            {"parameter": "r", "type": "absolute", "value": 0},
        ]
    },
    {
        "name": "rate_bump_+0.01",
        "shocks": [
            {"parameter": "r", "type": "absolute", "value": +0.01},
        ]
    },
    {
        "name": "rate_bump_+0.02",
        "shocks": [
            {"parameter": "r", "type": "absolute", "value": +0.02},
        ]
    },
    {
        "name": "rate_bump_+0.05",
        "shocks": [
            {"parameter": "r", "type": "absolute", "value": +0.05},
        ]
    },
]






def compute_scenario_prices(df: pd.DataFrame, scenario: Dict) -> pd.Series:
    """
    Core calculation: Apply shocks from a scenario to a DataFrame and return new prices.
    
    Args:
        df: Input DataFrame with columns ['S', 'K', 'T', 'r', 'sigma', 'option_type']
        scenario: Dictionary with "name" and "shocks" list.
        
    Returns:
        pd.Series: A series of option prices under the given scenario.
    """
    # Create local copies of parameters to apply shocks
    S = df['S'].copy()
    K = df['K'].copy()
    T = df['T'].copy()
    r = df['r'].copy()
    sigma = df['sigma'].copy()
    
    # 1. Apply each shock sequentially
    for shock in scenario.get("shocks", []):
        param = shock["parameter"]   # e.g., 'S', 'sigma'
        stype = shock["type"]        # 'relative' or 'absolute'
        val = shock["value"]
        
        # Mapping parameter names to local variables
        if param == 'S':
            S = S * (1 + val) if stype == "relative" else S + val
        elif param == 'sigma':
            sigma = sigma * (1 + val) if stype == "relative" else sigma + val
        elif param == 'r':
            r = r * (1 + val) if stype == "relative" else r + val
        elif param == 'K':
            K = K * (1 + val) if stype == "relative" else K + val
        elif param == 'T':
            T = T * (1 + val) if stype == "relative" else T + val

    # 2. Vectorized Pricing Logic (Black-Scholes)
    # Ensure sigma and T are positive to avoid math errors
    price = calc_bsm_price(S, K, T, r, sigma, df['option_type'])
    
    # 3. Return the price series based on option type
    return price



def run_sensitivity_analysis(
    input_path: str, 
    scenarios: List[Dict] | None = None
) -> str:
    """
    Main entry point for the Agent to perform sensitivity analysis on an entire CSV.
    
    Args:
        input_path (str): Path to the source option data.
        scenarios (List[Dict]): List of scenario configurations.
        
    Returns:
        str: Absolute path to the exported results file.
    """
    if scenarios is None:
        scenarios = DEFAULT_SENSITIVITY_SCENARIOS
        
    # Data loading
    df = pd.read_csv(input_path)
    result_df = df.copy()
    
    # Process each scenario by calling the core computation function
    for sc in scenarios:
        column_name = f"price_scen_{sc['name']}"
        result_df[column_name] = compute_scenario_prices(df, sc)
    
    # Persistence: Use the previously defined helper to save to local storage
    # _save_to_local handles directory creation and timestamping
    output_path = save_to_local(
        result_df, 
        folder_name="analytics", 
        prefix="stress_test_results"
    )
    
    return output_path