from typing import Union, Dict, Any, List, Annotated, Optional
import json
import pandas as pd
import os
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

DEFAULT_SCENARIOS = [
    {"name": "Black Monday (1987)", "spot_change": -0.20, "vol_change": 0.50, "rate_change": -0.005},
    {"name": "Dot-com Crash (2000)", "spot_change": -0.70, "vol_change": 2.00, "rate_change": -0.02},
    {"name": "2008 Financial Crisis", "spot_change": -0.50, "vol_change": 1.50, "rate_change": -0.01},
    {"name": "VIX Spike (No Stock Move)", "spot_change": 0.0, "vol_change": 1.00, "rate_change": 0.0},
    {"name": "Rate Shock (+200bps)", "spot_change": 0.0, "vol_change": 0.0, "rate_change": 0.02},
    {"name": "Liquidation Scenario", "spot_change": -0.30, "vol_change": 3.00, "rate_change": 0.01},
    {"name": "Volatility Collapse", "spot_change": 0.05, "vol_change": -0.50, "rate_change": 0.0},
]

# ============================================================================
# BLACK-SCHOLES PRICING ENGINE (Helper)
# ============================================================================

def _bsm_price(
    option_type: Union[str, pd.Series],
    S: Union[float, pd.Series],
    K: Union[float, pd.Series],
    T: Union[float, pd.Series],
    r: Union[float, pd.Series],
    sigma: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """
    Calculate Black-Scholes price for European option (Vectorized).
    """
    # Ensure T is not zero to avoid division by zero
    sqrtT = np.sqrt(np.maximum(T, 1e-9))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    
    # Vectorized conditions
    is_call = (option_type == 'call') if isinstance(option_type, str) else (option_type.str.lower() == 'call')
    
    price_call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    price_put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Handle T <= 0 case (intrinsic value)
    intrinsic_call = np.maximum(S - K, 0.0)
    intrinsic_put = np.maximum(K - S, 0.0)
    
    final_price = np.where(is_call, 
                           np.where(T <= 0, intrinsic_call, price_call),
                           np.where(T <= 0, intrinsic_put, price_put))
    
    return final_price

def _bsm_delta(
    option_type: Union[str, pd.Series],
    S: Union[float, pd.Series],
    K: Union[float, pd.Series],
    T: Union[float, pd.Series],
    r: Union[float, pd.Series],
    sigma: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """Calculate option delta (∂Price/∂S)."""
    sqrtT = np.sqrt(np.maximum(T, 1e-9))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    
    is_call = (option_type == 'call') if isinstance(option_type, str) else (option_type.str.lower() == 'call')
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1.0
    
    # Handle T <= 0 case
    intrinsic_delta_call = np.where(S > K, 1.0, 0.0)
    intrinsic_delta_put = np.where(S < K, -1.0, 0.0)
    
    return np.where(T <= 0, 
                    np.where(is_call, intrinsic_delta_call, intrinsic_delta_put),
                    np.where(is_call, delta_call, delta_put))

def _bsm_gamma(
    S: Union[float, pd.Series],
    K: Union[float, pd.Series],
    T: Union[float, pd.Series],
    r: Union[float, pd.Series],
    sigma: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """Calculate option gamma (∂²Price/∂S²)."""
    sqrtT = np.sqrt(np.maximum(T, 1e-9))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    gamma = norm.pdf(d1) / (S * sigma * sqrtT)
    return np.where(T <= 0, 0.0, gamma)

def _bsm_vega(
    S: Union[float, pd.Series],
    K: Union[float, pd.Series],
    T: Union[float, pd.Series],
    r: Union[float, pd.Series],
    sigma: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """Calculate option vega (∂Price/∂σ, per 1% change)."""
    sqrtT = np.sqrt(np.maximum(T, 1e-9))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    vega = S * norm.pdf(d1) * sqrtT / 100.0
    return np.where(T <= 0, 0.0, vega)

def _bsm_theta(
    option_type: Union[str, pd.Series],
    S: Union[float, pd.Series],
    K: Union[float, pd.Series],
    T: Union[float, pd.Series],
    r: Union[float, pd.Series],
    sigma: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """Calculate option theta (∂Price/∂T, per day)."""
    sqrtT = np.sqrt(np.maximum(T, 1e-9))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    
    theta_annual_base = (-S * norm.pdf(d1) * sigma / (2 * sqrtT))
    
    is_call = (option_type == 'call') if isinstance(option_type, str) else (option_type.str.lower() == 'call')
    theta_annual_call = theta_annual_base - r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_annual_put = theta_annual_base + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    theta_annual = np.where(is_call, theta_annual_call, theta_annual_put)
    return np.where(T <= 0, 0.0, theta_annual / 365.0)

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
# SENSITIVITY TEST
# ============================================================================

def _run_stress_test(
    df: pd.DataFrame,
    scenario: Dict,
) -> pd.DataFrame:
    """
    Run stress test with extreme market scenarios for multiple rows.
    Returns only the scenario-specific columns.
    """
    name = scenario.get('name', 'Unknown')
    spot_change = float(scenario.get('spot_change', 0.0))
    vol_change = float(scenario.get('vol_change', 0.0))
    rate_change = float(scenario.get('rate_change', 0.0))

    # Base parameters (as Series)
    opt_type = df['option_type']
    S = df['S']
    K = df['K']
    T = df['T']
    r = df['r']
    sigma = df['sigma']

    # Vectorized base price calculation (already calculated once in run_stress_test_to_file but for robustness)
    # Actually, we expect 'base_price' to be in df if we want to save recalculating, 
    # but for a pure function, we can take it from df or recalculate. 
    # Let's assume we pass it in.
    if 'base_price' in df.columns:
        base_price = df['base_price']
    else:
        base_price = _bsm_price(opt_type, S, K, T, r, sigma)

    # Calculate stressed parameters
    S_stressed = S * (1 + spot_change)
    sigma_stressed = np.maximum(0.001, sigma + vol_change)
    r_stressed = r + rate_change
    
    # Vectorized stressed price calculation
    stressed_price = _bsm_price(opt_type, S_stressed, K, T, r_stressed, sigma_stressed)
    
    pnl = stressed_price - base_price
    pnl_pct = np.where(base_price != 0, (pnl / base_price * 100), 0.0)
    
    # Create result DataFrame with scenario columns
    results = pd.DataFrame(index=df.index)
    results[f"{name}_spot_change_pct"] = spot_change * 100
    results[f"{name}_vol_change_pct"] = vol_change * 100
    results[f"{name}_rate_change_pct"] = rate_change * 100
    results[f"{name}_stressed_price"] = stressed_price
    results[f"{name}_PnL"] = pnl
    results[f"{name}_PnL%"] = pnl_pct
    
    return results

def run_stress_test_to_file(
    input_path: str, output_dir: str,
    scenarios: Optional[str] = DEFAULT_SCENARIOS
) -> str:
    """
    Run stress tests for all options in CSV using extreme market scenarios.
    
    Args:
        input_path: Path to CSV with option parameters
        output_dir: Directory to save results
    
    Returns:
        Absolute path to the resulting CSV file
    """
    stress_scenarios = scenarios

    try:
        if not os.path.exists(input_path):
            return f"Error: Input file not found at {input_path}"
        
        df = pd.read_csv(input_path)
        if df.empty:
            return "Error: CSV is empty"
        
        # skip duplicate rows as requested
        df = df.drop_duplicates().reset_index(drop=True)
        
        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return f"Error: Missing columns: {missing}"
        
        # Pre-calculate base price once
        df['base_price'] = _bsm_price(df['option_type'], df['S'], df['K'], df['T'], df['r'], df['sigma'])

        pnl_cols = []
        scenario_results = []
        for scenario in stress_scenarios:
            res_df = _run_stress_test(df, scenario)
            scenario_results.append(res_df)
            pnl_cols.append(f"{scenario['name']}_PnL")
        
        # Merge all results horizontally
        final_df = pd.concat([df] + scenario_results, axis=1)
        
        # Identify worst-case scenario per row
        # idxmin skips NaNs by default. If all are NaN, it returns NaN.
        final_df['worst_case_col'] = final_df[pnl_cols].idxmin(axis=1)
        final_df['worst_case_pnl'] = final_df[pnl_cols].min(axis=1)
        
        # Handle scenario name extraction safely
        def get_worst_scenario_name(col_val):
            if pd.isna(col_val):
                return "None"
            return str(col_val).replace("_PnL", "")

        final_df['worst_case_scenario'] = final_df['worst_case_col'].apply(get_worst_scenario_name)
        
        # Get worst case PnL% safely
        def get_worst_pnl_pct(row):
            scen = row['worst_case_scenario']
            if scen == "None":
                return np.nan
            col_name = f"{scen}_PnL%"
            return row.get(col_name, np.nan)

        final_df['worst_case_pnl_pct'] = final_df.apply(get_worst_pnl_pct, axis=1)
        
        # Clean up temporary column
        final_df = final_df.drop(columns=['worst_case_col'])
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", f"_stress_test.csv")
        output_path = os.path.join(output_dir, filename)
        
        final_df.to_csv(output_path, index=False)
        
        return os.path.abspath(output_path)
    
    except Exception as e:
        return f"Error: Stress test execution failed: {str(e)}"

# ============================================================================
# P&L ANALYSIS TEST (Wraps run_pnl_test from tools)
# ============================================================================

def _calculate_scenario_pnl(
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
    S0 = float(row['S'])
    K = float(row['K'])
    T0 = float(row['T'])
    r = float(row['r'])
    sigma0 = float(row['sigma'])

    # Base case
    V0 = _bsm_price(opt_type, S0, K, T0, r, sigma0)
    delta0 = _bsm_delta(opt_type, S0, K, T0, r, sigma0)
    gamma0 = _bsm_gamma(S0, K, T0, r, sigma0)
    vega0 = _bsm_vega(S0, K, T0, r, sigma0)
    theta0 = _bsm_theta(opt_type, S0, K, T0, r, sigma0)

    scenario_name = scenario.get('name', 'Unknown')
    spot_change = scenario.get('spot_change', 0.0)
    vol_change = scenario.get('vol_change', 0.0)
    days_passed = scenario.get('days_passed', 0.0)
    rate_change = scenario.get('rate_change', 0.0)

    S1 = S0*(1+spot_change)
    sigma1 = sigma0*(1+vol_change)
    T1 = max(0, T0 - days_passed / 365.0)
    r1 = r*(1+rate_change)

    # New option price
    V1 = _bsm_price(opt_type, S1, K, T1, r1, sigma1)
    actual_pnl = V1 - V0

    # Greeks-based P&L estimate
    spot_move = S1 - S0
    vol_move = sigma1 - sigma0
    rate_move = r1 - r
    time_decay = days_passed / 365.0

    # P&L components
    delta_pnl = delta0 * spot_move
    gamma_pnl = 0.5 * gamma0 * (spot_move ** 2)
    vega_pnl = vega0 * vol_move
    theta_pnl = theta0 * time_decay

    # Total estimated P&L
    estimated_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    pnl_error = actual_pnl - estimated_pnl

    # Realized variance (gamma P&L proxy)
    realized_var = (spot_move / S0) ** 2 if S0 != 0 else 0

    # Delta-hedged P&L (excluding delta component)
    hedged_pnl = gamma_pnl + vega_pnl + theta_pnl

    result = {
        "scenario_name": scenario_name,
        "spot_level": S1,
        "spot_move": spot_move,
        "vol_level": sigma1,
        "vol_move": vol_move,
        "days_passed": days_passed,
        "rate_level": r1,
        "rate_move": rate_move,
        "base_price": V0,
        "new_price": V1,
        "actual_pnl": actual_pnl,
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

def _summarize_pnl_scenarios(
    row: pd.Series,
    scenarios: List[Dict] | None = None,
) -> pd.Series:
    """
    Run multiple P&L scenarios for a single option and calculate summary metrics.

    Aggregates results across all provided scenarios to find average/max/min P&L
    and errors to evaluate pricing model stability or hedging effectiveness.

    Args:
        row (pd.Series): A row containing option parameters.
        scenarios (List[Dict] | None): A list of scenario dictionaries. Defaults to global DEFAULT_SCENARIOS.

    Returns:
        pd.Series: Summary statistics including num_scenarios, avg_actual_pnl, max_pnl_error, etc.
    """
    if not scenarios:
        # Default market moves for demonstration
        scenarios = DEFAULT_SCENARIOS
    
    details = [_calculate_scenario_pnl(row, s) for s in scenarios]

    num_scenarios = len(details)
    avg_actual_pnl = np.mean([d['actual_pnl'] for d in details])
    max_actual_pnl = np.max([d['actual_pnl'] for d in details])
    min_actual_pnl = np.min([d['actual_pnl'] for d in details])
    avg_pnl_error = np.mean([d['pnl_error'] for d in details])
    max_pnl_error = np.max(np.abs([d['pnl_error'] for d in details]))
    avg_delta_hedged_pnl = np.mean([d['delta_hedged_pnl'] for d in details])
    
    
    return pd.Series({
        'num_scenarios': num_scenarios,
        'avg_actual_pnl': avg_actual_pnl,
        'max_actual_pnl': max_actual_pnl,
        'min_actual_pnl': min_actual_pnl,
        'avg_pnl_error': avg_pnl_error,
        'max_pnl_error': max_pnl_error,
        'avg_delta_hedged_pnl': avg_delta_hedged_pnl,
        'details': details
    })

def run_pnl_attribution_test_to_file(
    input_path: str, 
    output_dir: str,
    scenarios: List[Dict] | None = None,
) -> str:
    """
    Execute P&L attribution tests for all options in a CSV file.

    Reads option data, runs specified scenarios for each option, aggregates results,
    and saves the detailed analysis to a CSV file.

    Args:
        input_path (str): Absolute path to the input CSV file containing option data.
        output_dir (str): Directory where the output CSV will be saved.
        scenarios (List[Dict] | None): Optional list of scenarios to test. Defaults to DEFAULT.

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
        
        if not scenarios:
            # Default market moves for demonstration
            scenarios = DEFAULT_SCENARIOS
        
        results = df.apply(lambda r: _summarize_pnl_scenarios(r, scenarios), axis=1)
        summary_df = pd.concat([df, results], axis=1)
        
        filename = os.path.basename(input_path).replace(".csv", "_pnl_test_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        summary_df.to_csv(output_path, index=False)
        
        return os.path.abspath(output_path)
    
    except Exception as e:
        return f"P&L test error: {str(e)}"
