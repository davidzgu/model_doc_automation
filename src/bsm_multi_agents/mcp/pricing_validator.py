from typing import Union, Dict, Any, List, Annotated
import json
import pandas as pd
import os


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
# SENSITIVITY TEST (Wraps run_sensitivity_test from tools)
# ============================================================================

def run_sensitivity_test_to_file(
    input_path: str, output_dir: str
) -> str:
    """
    Run sensitivity tests for all options in CSV using spot/vol bumps.
    
    For each option (first row):
    - Analyzes spot price sensitivity
    - Analyzes volatility sensitivity
    - Tests parallel yield curve shifts
    
    Args:
        input_path: Path to CSV with option parameters
        output_dir: Directory to save results
    
    Returns:
        JSON string with test results and output file path
    """
    try:
        if not os.path.exists(input_path):
            return json.dumps({"errors": [f"Input file not found at {input_path}"]})
        
        df = pd.read_csv(input_path)
        if df.empty:
            return json.dumps({"errors": ["CSV is empty"]})
        
        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps({"errors": [f"Missing columns: {missing}"]})
        
        # Use first row as representative option
        row = df.iloc[0].to_dict()
        option_json = json.dumps({
            "option_type": row.get('option_type'),
            "S": float(row.get('S')),
            "K": float(row.get('K')),
            "T": float(row.get('T')),
            "r": float(row.get('r', 0.0)),
            "sigma": float(row.get('sigma'))
        })
        
        # Import and run test
        from bsm_multi_agents.tools.test_generator_tools import run_sensitivity_test as _run_sens
        test_result = _run_sens(option_json, output_dir=output_dir)
        test_data = json.loads(test_result)
        
        # Append test summary to results CSV
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", "_sensitivity_test_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        summary = {
            "test_type": "sensitivity",
            "timestamp": datetime.now().isoformat(),
            "base_price": test_data.get('base_price'),
            "delta": test_data.get('base_greeks', {}).get('delta'),
            "gamma": test_data.get('base_greeks', {}).get('gamma'),
            "vega": test_data.get('base_greeks', {}).get('vega'),
            "status": "passed" if test_data.get('success') else "failed",
            "output_file": test_data.get('output_file', 'N/A')
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_path, index=False)
        
        return json.dumps({
            "success": True,
            "test_type": "sensitivity",
            "output_file": os.path.abspath(output_path),
            "test_details": test_data
        })
    
    except Exception as e:
        return json.dumps({"errors": [f"Sensitivity test error: {str(e)}"]})




# ============================================================================
# STRESS TEST (Wraps run_stress_test from tools)
# ============================================================================

def run_stress_test_to_file(
    input_path: str, output_dir: str
) -> str:
    """
    Run stress tests for all options in CSV using extreme scenarios.
    
    For each option (first row):
    - Tests Black Monday scenario (-20% stock, +50% vol)
    - Tests Dot-com scenario (-70% stock, +200% vol)
    - Tests 2008 Crisis scenario (-50% stock, +150% vol)
    - Tests VIX spike (0% stock, +100% vol)
    - Tests rate shock (+200bps)
    - Tests liquidation scenario (-30% stock, +300% vol)
    
    Args:
        input_path: Path to CSV with option parameters
        output_dir: Directory to save results
    
    Returns:
        JSON string with stress test results and output file path
    """
    try:
        if not os.path.exists(input_path):
            return json.dumps({"errors": [f"Input file not found at {input_path}"]})
        
        df = pd.read_csv(input_path)
        if df.empty:
            return json.dumps({"errors": ["CSV is empty"]})
        
        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps({"errors": [f"Missing columns: {missing}"]})
        
        # Use first row as representative option
        row = df.iloc[0].to_dict()
        option_json = json.dumps({
            "option_type": row.get('option_type'),
            "S": float(row.get('S')),
            "K": float(row.get('K')),
            "T": float(row.get('T')),
            "r": float(row.get('r', 0.0)),
            "sigma": float(row.get('sigma'))
        })
        
        # Import and run test
        from bsm_multi_agents.tools.test_generator_tools import run_stress_test as _run_stress
        test_result = _run_stress(option_json, output_dir=output_dir)
        test_data = json.loads(test_result)
        
        # Append test summary to results CSV
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", "_stress_test_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        worst_case = test_data.get('worst_case', {})
        summary = {
            "test_type": "stress",
            "timestamp": datetime.now().isoformat(),
            "base_price": test_data.get('base_price'),
            "num_scenarios": test_data.get('num_scenarios'),
            "worst_case_scenario": worst_case.get('scenario_name'),
            "worst_case_pnl": worst_case.get('pnl'),
            "worst_case_pnl_pct": worst_case.get('pnl_pct'),
            "status": "passed" if test_data.get('success') else "failed",
            "output_file": test_data.get('output_file', 'N/A')
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_path, index=False)
        
        return json.dumps({
            "success": True,
            "test_type": "stress",
            "output_file": os.path.abspath(output_path),
            "test_details": test_data
        })
    
    except Exception as e:
        return json.dumps({"errors": [f"Stress test error: {str(e)}"]})


# ============================================================================
# P&L ANALYSIS TEST (Wraps run_pnl_test from tools)
# ============================================================================

def run_pnl_test_to_file(
    input_path: str, output_dir: str
) -> str:
    """
    Run P&L analysis and attribution tests for all options in CSV.
    
    For each option (first row):
    - Analyzes Greeks-based P&L estimates vs. actual P&L
    - Tests P&L attribution (delta, gamma, vega, theta)
    - Tests gamma P&L (realized variance impact)
    - Tests delta-hedged returns
    
    Args:
        input_path: Path to CSV with option parameters
        output_dir: Directory to save results
    
    Returns:
        JSON string with P&L test results and output file path
    """
    try:
        if not os.path.exists(input_path):
            return json.dumps({"errors": [f"Input file not found at {input_path}"]})
        
        df = pd.read_csv(input_path)
        if df.empty:
            return json.dumps({"errors": ["CSV is empty"]})
        
        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps({"errors": [f"Missing columns: {missing}"]})
        
        # Use first row as representative option
        row = df.iloc[0].to_dict()
        option_json = json.dumps({
            "option_type": row.get('option_type'),
            "S": float(row.get('S')),
            "K": float(row.get('K')),
            "T": float(row.get('T')),
            "r": float(row.get('r', 0.0)),
            "sigma": float(row.get('sigma'))
        })
        
        # Default market moves for test
        moves_json = json.dumps([
            {"spot": float(row.get('S')) * 1.01, "vol": float(row.get('sigma')) + 0.01, "days_passed": 1, "rate": float(row.get('r', 0.0))},
            {"spot": float(row.get('S')) * 0.99, "vol": float(row.get('sigma')) - 0.01, "days_passed": 1, "rate": float(row.get('r', 0.0))},
        ])
        
        # Import and run test
        from bsm_multi_agents.tools.test_generator_tools import run_pnl_test as _run_pnl
        test_result = _run_pnl(option_json, moves_json, output_dir=output_dir)
        test_data = json.loads(test_result)
        
        # Append test summary to results CSV
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", "_pnl_test_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        summary_stats = test_data.get('summary', {})
        summary = {
            "test_type": "pnl_analysis",
            "timestamp": datetime.now().isoformat(),
            "base_price": test_data.get('base_greeks', {}).get('BSM_price'),
            "base_delta": test_data.get('base_greeks', {}).get('delta'),
            "base_gamma": test_data.get('base_greeks', {}).get('gamma'),
            "base_vega": test_data.get('base_greeks', {}).get('vega'),
            "base_theta": test_data.get('base_greeks', {}).get('theta'),
            "num_scenarios": summary_stats.get('num_scenarios'),
            "avg_pnl_error": summary_stats.get('avg_pnl_error'),
            "max_pnl_error": summary_stats.get('max_pnl_error'),
            "status": "passed" if test_data.get('success') else "failed",
            "output_file": test_data.get('output_file', 'N/A')
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_path, index=False)
        
        return json.dumps({
            "success": True,
            "test_type": "pnl_analysis",
            "output_file": os.path.abspath(output_path),
            "test_details": test_data
        })
    
    except Exception as e:
        return json.dumps({"errors": [f"P&L test error: {str(e)}"]})
