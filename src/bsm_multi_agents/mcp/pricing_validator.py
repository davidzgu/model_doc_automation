from typing import Union, Dict, Any, List, Annotated, Optional
import json
import pandas as pd
import os
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta


DEFAULT_STRESS_SCENARIOS = [
    {"name": "Black Monday (1987)", "spot_change": -0.20, "vol_change": 0.50, "rate_change": -0.005},
    {"name": "Dot-com Crash (2000)", "spot_change": -0.70, "vol_change": 2.00, "rate_change": -0.02},
    {"name": "2008 Financial Crisis", "spot_change": -0.50, "vol_change": 1.50, "rate_change": -0.01},
    {"name": "VIX Spike (No Stock Move)", "spot_change": 0.0, "vol_change": 1.00, "rate_change": 0.0},
    {"name": "Rate Shock (+200bps)", "spot_change": 0.0, "vol_change": 0.0, "rate_change": 0.02},
    {"name": "Liquidation Scenario", "spot_change": -0.30, "vol_change": 3.00, "rate_change": 0.01},
    {"name": "Volatility Collapse", "spot_change": 0.05, "vol_change": -0.50, "rate_change": 0.0},
]


DEFAULT_SENSITIVITY_SCENARIOS = [
    {"name": "spot_bump_-0.05", "spot_change": -0.05},
    {"name": "spot_bump_-0.02", "spot_change": -0.02},
    {"name": "spot_bump_-0.01", "spot_change": -0.01},
    {"name": "spot_bump_0", "spot_change": 0.0},
    {"name": "spot_bump_0.01", "spot_change": 0.01},
    {"name": "spot_bump_0.02", "spot_change": 0.02},
    {"name": "spot_bump_0.05", "spot_change": 0.05},
    {"name": "vol_bump_-0.05", "vol_change": -0.05},
    {"name": "vol_bump_-0.02", "vol_change": -0.02},    
    {"name": "vol_bump_-0.01", "vol_change": -0.01},
    {"name": "vol_bump_0", "vol_change": 0.0},
    {"name": "vol_bump_0.01", "vol_change": 0.01},
    {"name": "vol_bump_0.02", "vol_change": 0.02},
    {"name": "vol_bump_0.05", "vol_change": 0.05},
    {"name": "rate_bump_-0.05", "rate_change": -0.05},
    {"name": "rate_bump_-0.02", "rate_change": -0.02},
    {"name": "rate_bump_-0.01", "rate_change": -0.01},
    {"name": "rate_bump_0", "rate_change": 0.0},
    {"name": "rate_bump_0.01", "rate_change": 0.01},
    {"name": "rate_bump_0.02", "rate_change": 0.02},
    {"name": "rate_bump_0.05", "rate_change": 0.05},
]


def _validate_greeks_rules(
        option_type: str,
        BSM_price: float, 
        delta: float, 
        gamma: float, 
        vega: float
    ) -> str:
    """
    Validate Greeks against business rules.
    Returns list of validation results.
    """
    validations_result = "passed"
    validations_details = []

    # Rule 1: BSM_price > 0
    try:
        price_valid = BSM_price > 0
        if not price_valid:
            validations_result = "failed"
            validations_details.append(f"BSM_price {BSM_price} is not positive")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"BSM_price Error: {str(e)}")

    # Rule 2: delta range
    try:
        if option_type.lower() == 'call':
            delta_valid = 0 <= delta <= 1
            expected_range = "[0, 1]"
        else:
            delta_valid = -1 <= delta <= 0
            expected_range = "[-1, 0]"
        if not delta_valid:
            validations_result = "failed"
            validations_details.append(f"Delta {delta:.4f} outside {expected_range}")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Delta Error: {str(e)}")

    # Rule 3: gamma >= 0
    try:
        gamma_valid = gamma >= 0
        if not gamma_valid:
            validations_result = "failed"
            validations_details.append(f"Gamma {gamma:.4f} is negative")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Gamma Error: {str(e)}")

    # Rule 4: vega >= 0
    try:
        vega_valid = vega >= 0
        if not gamma_valid:
            validations_result = "failed"
            validations_details.append(f"Vega {vega:.4f} is negative")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Vega Error: {str(e)}")

    results = {
        "validations_result": validations_result,
        "validations_details": validations_details
    }
    return results

# ============================================================================
# Simple TEST
# ============================================================================

def validate_greeks_to_file(
    input_path: str, output_dir: str
) -> str:
    """
    Validate Greeks for ALL options from CSV data.

    For each option:
    - Validates: BSM_price > 0
    - Validates: delta in [0,1] for calls, [-1,0] for puts
    - Validates: gamma >= 0, vega >= 0

    Args:
        state: InjectedState, state from the workflow, which contains csv_data


    Returns:
        JSON string containing validate_results
    """
    try:
        if not os.path.exists(input_path):
            return f"Error: Input file not found at {input_path}"

        df = pd.read_csv(input_path)
        required_cols = ['option_type', 'BSM_price', 'delta', 'gamma', 'vega']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps({"errors": [f"Missing columns: {missing}"]})

        def calc_row(row):
            res = _validate_greeks_rules(
                row['option_type'], 
                row['BSM_price'], 
                row['delta'], 
                row['gamma'], 
                row['vega'], 
            )
            return res
        
        expanded = df.apply(calc_row, axis=1).apply(pd.Series)
        result_cols = ['validations_result','validations_details']
        for col in result_cols:
            if col not in expanded:
                expanded[col] = pd.NA
        df = pd.concat([df, expanded[result_cols]], axis=1)
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", "_validate_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        df.to_csv(output_path, index=False)
        
        return os.path.abspath(output_path)

    except Exception as e:
        return json.dumps({"errors": [f"Error: {str(e)}"]})



# ============================================================================
# Sensitivity TEST 
# ============================================================================

def _calculate_scenario_attribution(
    row: pd.Series,
    scenario: Dict[str, float],
) -> Dict[str, float]:
    """
    Calculate P&L and Greek attribution for a single market scenario.

    Computes the actual P&L (based on BSM pricing) and compares it with
    the estimated P&L derived from Greeks (Delta, Gamma, Vega, Theta).

    Args:
        row (pd.Series): A row containing option parameters (S, K, T, r, sigma, option_type).
        scenario (Dict[str, float]): A dictionary defining the market move (spot_change, vol_change, etc.).

    Returns:
        Dict[str, float]: A dictionary containing actual P&L, estimated P&L, attribution components, and error metrics.
    """
    opt_type = row['option_type']
    S_base = float(row['S'])
    K_base = float(row['K'])
    T_base = float(row['T'])
    r_base = float(row['r'])
    sigma_base = float(row['sigma'])

    # Base case
    V_base = _bsm_price(opt_type, S_base, K_base, T_base, r_base, sigma_base)
    delta_base = _bsm_delta(opt_type, S_base, K_base, T_base, r_base, sigma_base)
    gamma_base = _bsm_gamma(S_base, K_base, T_base, r_base, sigma_base)
    vega_base = _bsm_vega(S_base, K_base, T_base, r_base, sigma_base)
    theta_base = _bsm_theta(opt_type, S_base, K_base, T_base, r_base, sigma_base)

    scenario_name = scenario.get('name', 'Unknown')
    spot_change = scenario.get('spot_change', 0.0)
    vol_change = scenario.get('vol_change', 0.0)
    days_passed = scenario.get('days_passed', 0.0)
    rate_change = scenario.get('rate_change', 0.0)

    S_stress = S_base*(1+spot_change)
    sigma_stress = np.maximum(0.001, sigma_base + vol_change)
    T_stress = max(0, T_base - days_passed / 365.0)
    r_stress = r_base + rate_change

    # New option price
    V_stress = _bsm_price(opt_type, S_stress, K_base, T_stress, r_stress, sigma_stress)
    actual_pnl = V_stress - V_base
    pnl_pct = np.where(V_base != 0, (actual_pnl / V_base * 100), 0.0)

    # Greeks-based P&L estimate
    spot_move = S_stress - S_base
    vol_move = sigma_stress - sigma_base
    rate_move = r_stress - r_base
    time_decay = days_passed / 365.0

    # P&L components
    delta_pnl = delta_base * spot_move
    gamma_pnl = 0.5 * gamma_base * (spot_move ** 2)
    vega_pnl = vega_base * vol_move
    theta_pnl = theta_base * time_decay

    # Total estimated P&L
    estimated_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    pnl_error = actual_pnl - estimated_pnl

    # Realized variance (gamma P&L proxy)
    realized_var = (spot_move / S_base) ** 2 if S_base != 0 else 0

    # Delta-hedged P&L (excluding delta component)
    hedged_pnl = gamma_pnl + vega_pnl + theta_pnl

    result = {
        # Scenario Info
        "scenario_name": scenario_name,
        "scen_spot_change": spot_change,
        "scen_vol_change": vol_change,
        "scen_time_step": days_passed, 
        "scen_rate_change": rate_change,
        
        # Prices
        "base_price": V_base,
        "new_price": V_stress,
        
        # PnL Attribution
        "actual_pnl": actual_pnl,
        "pnl_pct": pnl_pct,
        "delta_pnl": delta_pnl,
        "gamma_pnl": gamma_pnl,
        "vega_pnl": vega_pnl,
        "theta_pnl": theta_pnl,
        "estimated_pnl": estimated_pnl,
        "pnl_error": pnl_error,
        "realized_variance": realized_var,
        "delta_hedged_pnl": hedged_pnl
    }

    return result

def _run_sensitivity_test_on_single_trade(
    row: pd.Series,
    scenarios: List[Dict] = DEFAULT_SENSITIVITY_SCENARIOS,
) -> List[Dict]:
    """
    Run sensitivity analysis for a SINGLE trade across multiple scenarios.
    Returns a list of dictionaries, where each dictionary is a flattened row 
    containing BOTH the original trade info and the scenario results.
    """
    results = []
    
    # Convert original trade row to dict to preserve its info
    trade_info = row.to_dict()
    
    for sc in scenarios:
        # Calculate PnL/Attribution for this scenario
        attribution = _calculate_scenario_attribution(row, sc)
        
        # Merge: Trade Info + Scenario Results
        # Note: If keys collide (unlikely given naming), attribution overwrites trade_info.
        # usually trade_info has 'S', 'vol', etc. attribution has 'scen_spot_change', etc.
        combined_record = {**trade_info, **attribution}
        results.append(combined_record)
    
    return results

def run_sensitivity_test_to_file(
    input_path: str, 
    output_dir: str,
    scenarios: List[Dict] = DEFAULT_SENSITIVITY_SCENARIOS,
) -> str:
    """
    Execute P&L attribution tests for all options in a CSV file.
    Output is in LONG format: One row per trade per scenario.

    Args:
        input_path (str): Absolute path to the input CSV file containing option data.
        output_dir (str): Directory where the output CSV will be saved.
        scenarios List[Dict]: List of scenarios to test. Defaults to DEFAULT_SENSITIVITY_SCENARIOS.

    Returns:
        str: The absolute path to the generated result file.
    """
    try:
        if not os.path.exists(input_path):
            return f"Input file not found at {input_path}"
        
        df = pd.read_csv(input_path)
        if df.empty:
            return f"CSV is empty"
        
        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return f"Missing columns: {missing}"
        
        # Iterate over all trades calculate results
        all_results = []
        for _, row in df.iterrows():
            trade_results = _run_sensitivity_test_on_single_trade(row, scenarios)
            all_results.extend(trade_results)
            
        # Create final DataFrame from list of dicts
        final_df = pd.DataFrame(all_results)
        
        filename = os.path.basename(input_path).replace(".csv", "_sensitivity_test_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        # Save without index (row number is meaningless in long format)
        os.makedirs(output_dir, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        return os.path.abspath(output_path)
    
    except Exception as e:
        return f"P&L test error: {str(e)}"


# ============================================================================
# GAMMA POSITIVE TEST 
# ============================================================================


def _run_gamma_positivity_test(row: pd.Series) -> bool:
    """
    Check if the Gamma of an option is positive (convexity check).
    
    This function validates the fundamental property that long vanilla options 
    should have positive Gamma (convexity) by verifying that:
    V(S+h) + V(S-h) - 2V(S) > 0

    Args:
        row (pd.Series): A row containing option parameters:
            - 'option_type': 'call' or 'put'
            - 'S': Spot price
            - 'K': Strike price
            - 'T': Time to maturity
            - 'r': Risk-free rate
            - 'sigma': Volatility

    Returns:
        bool: True if Gamma is positive (convex), False otherwise.
    """

    opt_type = row['option_type']
    S0 = float(row['S'])
    K = float(row['K'])
    T0 = float(row['T'])
    r = float(row['r'])
    sigma0 = float(row['sigma'])

    V0 = _bsm_price(opt_type, S0, K, T0, r, sigma0)

    # Bump prices up and down by 1%
    bump = S0 * 0.01
    V_up = _bsm_price(opt_type, S0 + bump, K, T0, r, sigma0)
    V_down = _bsm_price(opt_type, S0 - bump, K, T0, r, sigma0)

    # Check gamma positivity condition
    gamma_condition = V_up + V_down - 2 * V0
    return gamma_condition > 0

def run_gamma_positivity_test_to_file(
    input_path: str, 
    output_dir: str,
) -> str:
    """
    Execute the specialized Gamma Positivity (Convexity) Test Suite.

    This is a distinct stress test that verifies the fundamental no-arbitrage
    condition of option convexity: V(S+h) + V(S-h) - 2V(S) > 0.
    It performs a bump-and-reprice analysis which is independent of the 
    standard Greek range validation.

    Args:
        input_path (str): Absolute path to the input CSV file containing option parameters.
        output_dir (str): Directory where the output CSV will be saved.

    Returns:
        str: Absolute path to the generated results file.
    """
    try:
        if not os.path.exists(input_path):
            return f"Input file not found at {input_path}"
        
        df = pd.read_csv(input_path)
        if df.empty:
            return f"CSV is empty"
        
        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return f"Missing columns: {missing}"
        
        results = df.apply(lambda r: _run_gamma_positivity_test(r), axis=1)
        summary_df = df.copy()
        summary_df['gamma_positivity'] = results
        
        filename = os.path.basename(input_path).replace(".csv", "_gamma_positivity_test_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        summary_df.to_csv(output_path, index=False)
        
        return os.path.abspath(output_path)
    
    except Exception as e:
        return f"Gamma test error: {str(e)}"



# ============================================================================
# SENSITIVITY TEST
# ============================================================================


# def run_stress_test_to_file(
#     input_path: str, output_dir: str,
#     scenarios: List[Dict] = DEFAULT_SCENARIOS
# ) -> str:
#     """
#     Run stress tests for all options in CSV using extreme market scenarios.
    
#     Args:
#         input_path: Path to CSV with option parameters
#         output_dir: Directory to save results
#         scenarios List[Dict]: List of scenarios to test. Defaults to DEFAULT_SCENARIOS.
    
#     Returns:
#         Absolute path to the resulting CSV file
#     """

#     try:
#         if not os.path.exists(input_path):
#             return f"Error: Input file not found at {input_path}"
        
#         df = pd.read_csv(input_path)
#         if df.empty:
#             return "Error: CSV is empty"
        
#         # skip duplicate rows as requested
#         df = df.drop_duplicates().reset_index(drop=True)
        
#         required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
#         missing = [c for c in required_cols if c not in df.columns]
#         if missing:
#             return f"Error: Missing columns: {missing}"
        
#         # Pre-calculate base price once
#         df['base_price'] = _bsm_price(df['option_type'], df['S'], df['K'], df['T'], df['r'], df['sigma'])

#         pnl_cols = []
#         scenario_results = []
#         for scenario in scenarios:
#             res_df = _run_stress_test(df, scenario)
#             scenario_results.append(res_df)
#             pnl_cols.append(f"{scenario['name']}_PnL")
        
#         # Merge all results horizontally
#         final_df = pd.concat([df] + scenario_results, axis=1)
        
#         # Identify worst-case scenario per row
#         # idxmin skips NaNs by default. If all are NaN, it returns NaN.
#         final_df['worst_case_col'] = final_df[pnl_cols].idxmin(axis=1)
#         final_df['worst_case_pnl'] = final_df[pnl_cols].min(axis=1)
        
#         # Handle scenario name extraction safely
#         def get_worst_scenario_name(col_val):
#             if pd.isna(col_val):
#                 return "None"
#             return str(col_val).replace("_PnL", "")

#         final_df['worst_case_scenario'] = final_df['worst_case_col'].apply(get_worst_scenario_name)
        
#         # Get worst case PnL% safely
#         def get_worst_pnl_pct(row):
#             scen = row['worst_case_scenario']
#             if scen == "None":
#                 return np.nan
#             col_name = f"{scen}_PnL%"
#             return row.get(col_name, np.nan)

#         final_df['worst_case_pnl_pct'] = final_df.apply(get_worst_pnl_pct, axis=1)
        
#         # Clean up temporary column
#         final_df = final_df.drop(columns=['worst_case_col'])
        
#         # Save results
#         os.makedirs(output_dir, exist_ok=True)
#         filename = os.path.basename(input_path).replace(".csv", f"_stress_test_results.csv")
#         output_path = os.path.join(output_dir, filename)
        
#         final_df.to_csv(output_path, index=False)
        
#         return os.path.abspath(output_path)
    
#     except Exception as e:
#         return f"Error: Stress test execution failed: {str(e)}"

# def _summarize_pnl_scenarios(
#     row: pd.Series,
#     scenarios: List[Dict] | None = None,
# ) -> pd.Series:
#     """
#     Run multiple P&L scenarios for a single option and calculate summary metrics.

#     Aggregates results across all provided scenarios to find average/max/min P&L
#     and errors to evaluate pricing model stability or hedging effectiveness.

#     Args:
#         row (pd.Series): A row containing option parameters.
#         scenarios (List[Dict] | None): A list of scenario dictionaries. Defaults to global DEFAULT_SCENARIOS.

#     Returns:
#         pd.Series: Summary statistics including num_scenarios, avg_actual_pnl, max_pnl_error, etc.
#     """
#     if not scenarios:
#         # Default market moves for demonstration
#         scenarios = DEFAULT_PNL_SCENARIOS
    
#     details = [_calculate_scenario_pnl(row, s) for s in scenarios]

#     num_scenarios = len(details)
#     avg_actual_pnl = np.mean([d['actual_pnl'] for d in details])
#     max_actual_pnl = np.max([d['actual_pnl'] for d in details])
#     min_actual_pnl = np.min([d['actual_pnl'] for d in details])
#     avg_pnl_error = np.mean([d['pnl_error'] for d in details])
#     max_pnl_error = np.max(np.abs([d['pnl_error'] for d in details]))
#     avg_delta_hedged_pnl = np.mean([d['delta_hedged_pnl'] for d in details])
    
    
#     return pd.Series({
#         'num_scenarios': num_scenarios,
#         'avg_actual_pnl': avg_actual_pnl,
#         'max_actual_pnl': max_actual_pnl,
#         'min_actual_pnl': min_actual_pnl,
#         'avg_pnl_error': avg_pnl_error,
#         'max_pnl_error': max_pnl_error,
#         'avg_delta_hedged_pnl': avg_delta_hedged_pnl,
#         'details': details
#     })


