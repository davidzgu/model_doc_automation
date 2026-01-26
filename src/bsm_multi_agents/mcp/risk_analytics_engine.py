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


DEFAULT_STRESS_SCENARIOS = [
    {
        "name": "Black Monday (1987)",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": -0.20},
            {"parameter": "sigma", "type": "absolute", "value": 0.50},
            {"parameter": "r", "type": "absolute", "value": -0.005}
        ]
    },
    {
        "name": "Dot-com Crash (2000)",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": -0.70},
            {"parameter": "sigma", "type": "absolute", "value": 2.00},
            {"parameter": "r", "type": "absolute", "value": -0.02}
        ]
    },
    {
        "name": "2008 Financial Crisis",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": -0.50},
            {"parameter": "sigma", "type": "absolute", "value": 1.50},
            {"parameter": "r", "type": "absolute", "value": -0.01}
        ]
    },
    {
        "name": "VIX Spike (No Stock Move)",
        "shocks": [
            {"parameter": "sigma", "type": "absolute", "value": 1.00}
        ]
    },
    {
        "name": "Rate Shock (+200bps)",
        "shocks": [
            {"parameter": "r", "type": "absolute", "value": 0.02}
        ]
    },
    {
        "name": "Liquidation Scenario",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": -0.30},
            {"parameter": "sigma", "type": "absolute", "value": 3.00},
            {"parameter": "r", "type": "absolute", "value": 0.01}
        ]
    },
    {
        "name": "Volatility Collapse",
        "shocks": [
            {"parameter": "S", "type": "relative", "value": 0.05},
            {"parameter": "sigma", "type": "absolute", "value": -0.50}
        ]
    }
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
        column_name = f"sensitivity_scen_{sc['name']}"
        result_df[column_name] = compute_scenario_prices(df, sc)
    
    # Persistence: Use the previously defined helper to save to local storage
    # _save_to_local handles directory creation and timestamping
    output_path = save_to_local(
        result_df, 
        folder_name="analytics", 
        prefix="sensitivity_test_results"
    )
    
    return output_path




def run_stress_analysis(
    input_path: str, 
    scenarios: List[Dict] | None = None
) -> str:
    """
    Main entry point for the Agent to perform stress analysis on an entire CSV.
    
    Args:
        input_path (str): Path to the source option data.
        scenarios (List[Dict]): List of scenario configurations.
        
    Returns:
        str: Absolute path to the exported results file.
    """
    if scenarios is None:
        scenarios = DEFAULT_STRESS_SCENARIOS
        
    # Data loading
    df = pd.read_csv(input_path)
    result_df = df.copy()
    
    # Process each scenario by calling the core computation function
    for sc in scenarios:
        column_name = f"stress_scen_{sc['name']}"
        result_df[column_name] = compute_scenario_prices(df, sc)
    
    # Persistence: Use the previously defined helper to save to local storage
    # _save_to_local handles directory creation and timestamping
    output_path = save_to_local(
        result_df, 
        folder_name="analytics", 
        prefix="stress_test_results"
    )
    
    return output_path

def _extract_shocks(df: pd.DataFrame, scenario: Dict) -> Dict[str, pd.Series]:
    """
    Internal helper to extract delta changes (shocks) for each parameter.
    Returns a dictionary of Series, e.g., {'S': ds_series, 'sigma': dvol_series}
    """
    shocks_applied = {
        'S': pd.Series(0.0, index=df.index),
        'sigma': pd.Series(0.0, index=df.index),
        'r': pd.Series(0.0, index=df.index),
        'T': pd.Series(0.0, index=df.index)
    }
    
    for shock in scenario.get("shocks", []):
        param = shock["parameter"]
        stype = shock["type"]
        val = shock["value"]
        
        if param in shocks_applied:
            if stype == "relative":
                # ds = S * 0.05
                shocks_applied[param] = df[param] * val
            else:
                # ds = 0.05 (absolute)
                shocks_applied[param] = pd.Series(val, index=df.index)
                
    return shocks_applied


def compute_pnl_attribution(df: pd.DataFrame, scenarios: List[Dict]) -> pd.DataFrame:
    """
    Decompose total P&L into Delta, Gamma, Vega, Theta, and Rho components.
    """
    res = df.copy()
    
    for scen in scenarios:
        scen_name = scen["name"]
        price_col = f"stress_scen_{scen_name}"
        
        if price_col not in res.columns:
            continue
            
        # 1. Calculate Actual P&L
        actual_pnl = res[price_col] - res['price']
        
        # 2. Extract Shocks for this scenario
        shocks = _extract_shocks(res, scen)
        ds = shocks['S']
        dv = shocks['sigma']
        dr = shocks['r']
        
        # 3. Attribution Calculation (Taylor Expansion)
        # PnL ~= Delta*dS + 0.5*Gamma*dS^2 + Vega*dV + Rho*dR + Theta*dT
        res[f'attr_delta_{scen_name}'] = res['delta'] * ds
        res[f'attr_gamma_{scen_name}'] = 0.5 * res['gamma'] * (ds ** 2)
        res[f'attr_vega_{scen_name}'] = res['vega'] * dv
        res[f'attr_rho_{scen_name}'] = res['rho'] * dr
        
        # 4. Total Explained and Residual
        explained = (res[f'attr_delta_{scen_name}'] + 
                     res[f'attr_gamma_{scen_name}'] + 
                     res[f'attr_vega_{scen_name}'] + 
                     res[f'attr_rho_{scen_name}'])
        
        res[f'attr_unexplained_{scen_name}'] = actual_pnl - explained
        res[f'attr_total_pnl_{scen_name}'] = actual_pnl

    return res

def PnL_attribution_analysis(input_path: str) -> str:
    """
    Perform P&L Attribution Analysis to explain the impact of Stress Testing scenarios.

    This tool takes the output from a stress test, calculates how much of the 
    price change in each extreme scenario was driven by specific risk factors 
    (Delta, Gamma, Vega, Rho, Theta), and identifies any unexplained residual P&L.

    Args:
        input_path (str): Absolute path to the CSV file generated by the 
            risk_analytics_engine. Must contain original Greeks and 'price_scen_*' columns.

    Returns:
        str: Absolute path to the resulting CSV file. The output includes attribution 
             columns for each scenario (e.g., 'attr_delta_Black_Monday', 'attr_vega_Black_Monday') 
             and a 'total_unexplained' column to verify Taylor expansion accuracy.

    Note:
        This is a critical step for risk reporting, allowing the Agent to explain 
        to the user NOT JUST that they lost money, but WHY (e.g., "90% of losses 
        were due to the Spot price drop in the 2008 Crisis scenario").
    """
    df = pd.read_csv(input_path)
    
    # Execute Attribution Logic
    analyzed_df = compute_pnl_attribution(df) # 建议使用之前改好的 compute 命名的函数
    
    # Save results
    return _save_to_local(analyzed_df, folder_name="outcome_analysis", prefix="pnl_attribution_report")