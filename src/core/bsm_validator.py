# -*- coding: utf-8 -*-
"""
BSM Validation Tools

Provides validation functions for Black-Scholes-Merton option pricing:
1. batch_greeks_validator - Validates Greeks for all options in dataset
2. validate_put_call_parity - Tests put-call parity for paired options
3. validate_sensitivity - Validates sensitivity analysis results

All validation tools are designed to work with CSV data from the data loader.
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List
from langchain_core.tools import tool
from scipy.stats import norm


# ============================================================================
# Core Calculation Functions (not tools, for internal use)
# ============================================================================

def _calculate_greeks(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """
    Internal function to calculate option Greeks.
    Returns dict with price, delta, gamma, vega, rho, theta.
    """
    option_type = option_type.lower()

    if T <= 0 or sigma <= 0:
        return {"error": "T and sigma must be positive"}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = float(norm.cdf(d1))
        rho = float(K * T * np.exp(-r * T) * norm.cdf(d2))
        theta = float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)))
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = float(norm.cdf(d1) - 1)
        rho = float(-K * T * np.exp(-r * T) * norm.cdf(-d2))
        theta = float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)))
    else:
        return {"error": "option_type must be 'call' or 'put'"}

    # Gamma and vega are same for call and put
    gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    vega = float(S * norm.pdf(d1) * np.sqrt(T))

    return {
        "price": float(price),
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "rho": rho,
        "theta": theta
    }


def _validate_greeks_rules(greeks: Dict[str, Any], option_type: str) -> List[Dict[str, Any]]:
    """
    Validate Greeks against business rules.
    Returns list of validation results.
    """
    validations = []

    # Rule 1: price > 0
    try:
        price_valid = greeks.get('price', 0) > 0
        validations.append({
            "rule": "price > 0",
            "status": "passed" if price_valid else "failed",
            "value": greeks.get('price'),
            "message": f"Price = {greeks.get('price'):.4f}" if price_valid else f"Price {greeks.get('price')} is not positive"
        })
    except Exception as e:
        validations.append({"rule": "price > 0", "status": "error", "message": str(e)})

    # Rule 2: delta range
    try:
        delta = greeks.get('delta', 0)
        if option_type.lower() == 'call':
            delta_valid = 0 <= delta <= 1
            expected_range = "[0, 1]"
        else:
            delta_valid = -1 <= delta <= 0
            expected_range = "[-1, 0]"

        validations.append({
            "rule": f"delta in {expected_range}",
            "status": "passed" if delta_valid else "failed",
            "value": delta,
            "message": f"Delta = {delta:.4f}" if delta_valid else f"Delta {delta:.4f} outside {expected_range}"
        })
    except Exception as e:
        validations.append({"rule": "delta range", "status": "error", "message": str(e)})

    # Rule 3: gamma >= 0
    try:
        gamma = greeks.get('gamma', -1)
        gamma_valid = gamma >= 0
        validations.append({
            "rule": "gamma >= 0",
            "status": "passed" if gamma_valid else "failed",
            "value": gamma,
            "message": f"Gamma = {gamma:.4f}" if gamma_valid else f"Gamma {gamma:.4f} is negative"
        })
    except Exception as e:
        validations.append({"rule": "gamma >= 0", "status": "error", "message": str(e)})

    # Rule 4: vega >= 0
    try:
        vega = greeks.get('vega', -1)
        vega_valid = vega >= 0
        validations.append({
            "rule": "vega >= 0",
            "status": "passed" if vega_valid else "failed",
            "value": vega,
            "message": f"Vega = {vega:.4f}" if vega_valid else f"Vega {vega:.4f} is negative"
        })
    except Exception as e:
        validations.append({"rule": "vega >= 0", "status": "error", "message": str(e)})

    return validations


def _parse_csv_data(csv_data: Union[str, Dict[str, Any]]) -> pd.DataFrame:
    """Parse CSV data into DataFrame."""

    # 🔬 调试探针 4
    print("\n" + "="*60)
    print("🔬 探针 4: _parse_csv_data 接收的数据")
    print("csv_data 类型:", type(csv_data))
    print("csv_data 内容:", csv_data)
    if isinstance(csv_data, dict) and 'option_type' in csv_data:
        print("option_type 子结构:", csv_data['option_type'])
        print("option_type 第一个值:", list(csv_data['option_type'].values())[0] if csv_data['option_type'] else "N/A")
    print("="*60 + "\n")


    if isinstance(csv_data, str):
        data = json.loads(csv_data)
    elif isinstance(csv_data, dict):
        data = csv_data
    else:
        raise ValueError(f"csv_data must be string or dict, got {type(csv_data)}")
    df = pd.DataFrame(data)

    # 🔬 调试探针 5
    print("\n" + "="*60)
    print("🔬 探针 5: DataFrame 创建后的结构")
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())
    print("DataFrame index:", df.index.tolist())
    print("DataFrame dtypes:")
    print(df.dtypes)
    print("\nDataFrame 前几行:")
    print(df.head())
    print("\noption_type 列的值:", df['option_type'].tolist())
    print("option_type 列的类型:", df['option_type'].dtype)
    print("="*60 + "\n")


    return df


# ============================================================================
# Validation Tools (LangChain tools for agents)
# ============================================================================

@tool
def batch_greeks_validator(csv_data: Union[str, Dict[str, Any]]) -> str:
    """
    Validate Greeks for ALL options in CSV data.

    For each option:
    - Calculates Greeks using its parameters
    - Validates: price > 0
    - Validates: delta in [0,1] for calls, [-1,0] for puts
    - Validates: gamma >= 0, vega >= 0

    Args:
        csv_data: CSV data with columns: option_type, S, K, T, r, sigma

    Returns:
        JSON with detailed validation results for each option
    """
    try:
        df = _parse_csv_data(csv_data)

        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps({"status": "error", "message": f"Missing columns: {missing}"})

        results = {
            "total_options": len(df),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "details": []
        }

        for idx, row in df.iterrows():
            # 🔬 调试探针 6
            print(f"\n🔬 探针 6: 处理第 {idx} 行")
            print(f"row 类型: {type(row)}")
            print(f"row 内容: {row.to_dict()}")
            print(f"row['option_type'] = {row['option_type']} (类型: {type(row['option_type'])})")
            
            option_type = str(row['option_type']).lower()
            S, K, T, r, sigma = float(row['S']), float(row['K']), float(row['T']), float(row['r']), float(row['sigma'])

            try:
                greeks = _calculate_greeks(option_type, S, K, T, r, sigma)

                if 'error' in greeks:
                    results['errors'] += 1
                    results['details'].append({
                        "option_index": int(idx),
                        "parameters": {"option_type": option_type, "S": S, "K": K, "T": T, "r": r, "sigma": sigma},
                        "calculation_error": greeks['error'],
                        "overall_status": "error"
                    })
                    continue

                validations = _validate_greeks_rules(greeks, option_type)
                all_passed = all(v['status'] == 'passed' for v in validations)
                has_error = any(v['status'] == 'error' for v in validations)

                if has_error:
                    overall_status = "error"
                    results['errors'] += 1
                elif all_passed:
                    overall_status = "passed"
                    results['passed'] += 1
                else:
                    overall_status = "failed"
                    results['failed'] += 1

                results['details'].append({
                    "option_index": int(idx),
                    "parameters": {"option_type": option_type, "S": S, "K": K, "T": T, "r": r, "sigma": sigma},
                    "greeks": greeks,
                    "validations": validations,
                    "overall_status": overall_status
                })

            except Exception as e:
                results['errors'] += 1
                results['details'].append({
                    "option_index": int(idx),
                    "parameters": {"option_type": option_type, "S": S, "K": K, "T": T, "r": r, "sigma": sigma},
                    "calculation_error": str(e),
                    "overall_status": "error"
                })

        results['status'] = 'error' if results['errors'] > 0 else ('failed' if results['failed'] > 0 else 'passed')
        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error: {str(e)}"})


@tool
def validate_put_call_parity(csv_data: Union[str, Dict[str, Any]]) -> str:
    """
    Test put-call parity for paired options.

    Finds call/put pairs with matching S, K, T, r, sigma and validates:
    C - P ≈ S - K*e^(-rT)

    Args:
        csv_data: CSV data with option parameters

    Returns:
        JSON with parity test results for all pairs
    """
    try:
        df = _parse_csv_data(csv_data)

        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps({"status": "error", "message": f"Missing columns: {missing}"})

        results = {
            "pairs_found": 0,
            "pairs_tested": 0,
            "pairs_passed": 0,
            "pairs_failed": 0,
            "details": []
        }

        tolerance = 1e-4

        for i, row_i in df.iterrows():
            for j, row_j in df.iterrows():
                if i >= j:
                    continue

                # Check params match (except option_type)
                params_match = (
                    abs(row_i['S'] - row_j['S']) < 1e-10 and
                    abs(row_i['K'] - row_j['K']) < 1e-10 and
                    abs(row_i['T'] - row_j['T']) < 1e-10 and
                    abs(row_i['r'] - row_j['r']) < 1e-10 and
                    abs(row_i['sigma'] - row_j['sigma']) < 1e-10
                )

                types_differ = str(row_i['option_type']).lower() != str(row_j['option_type']).lower()

                if params_match and types_differ:
                    results['pairs_found'] += 1

                    # Determine call and put
                    if str(row_i['option_type']).lower() == 'call':
                        call_idx, put_idx = i, j
                        call_row, put_row = row_i, row_j
                    else:
                        call_idx, put_idx = j, i
                        call_row, put_row = row_j, row_i

                    try:
                        S, K, T, r, sigma = (
                            float(call_row['S']), float(call_row['K']),
                            float(call_row['T']), float(call_row['r']),
                            float(call_row['sigma'])
                        )

                        call_greeks = _calculate_greeks('call', S, K, T, r, sigma)
                        put_greeks = _calculate_greeks('put', S, K, T, r, sigma)

                        if 'error' in call_greeks or 'error' in put_greeks:
                            results['details'].append({
                                "call_index": int(call_idx),
                                "put_index": int(put_idx),
                                "status": "error",
                                "message": "Error calculating Greeks"
                            })
                            continue

                        results['pairs_tested'] += 1

                        C, P = call_greeks['price'], put_greeks['price']
                        lhs = C - P
                        rhs = S - K * np.exp(-r * T)
                        diff = abs(lhs - rhs)

                        parity_holds = diff < tolerance

                        if parity_holds:
                            results['pairs_passed'] += 1
                            status = "passed"
                            msg = f"Parity holds: |{lhs:.4f} - {rhs:.4f}| = {diff:.6f}"
                        else:
                            results['pairs_failed'] += 1
                            status = "failed"
                            msg = f"Parity violated: |{lhs:.4f} - {rhs:.4f}| = {diff:.6f}"

                        results['details'].append({
                            "call_index": int(call_idx),
                            "put_index": int(put_idx),
                            "parameters": {"S": S, "K": K, "T": T, "r": r, "sigma": sigma},
                            "call_price": C,
                            "put_price": P,
                            "lhs": lhs,
                            "rhs": rhs,
                            "difference": diff,
                            "status": status,
                            "message": msg
                        })

                    except Exception as e:
                        results['details'].append({
                            "call_index": int(call_idx),
                            "put_index": int(put_idx),
                            "status": "error",
                            "message": f"Error: {str(e)}"
                        })

        if results['pairs_found'] == 0:
            results['status'] = 'skipped'
            results['message'] = 'No call/put pairs found'
        elif results['pairs_failed'] > 0:
            results['status'] = 'failed'
        else:
            results['status'] = 'passed'

        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error: {str(e)}"})


@tool
def validate_sensitivity(csv_data: Union[str, Dict[str, Any]]) -> str:
    """
    Run sensitivity analysis on first option and validate results.

    Tests spot price sensitivity from -2.5% to +2.5% (11 points).
    Validates all entries have required fields and no errors.

    Args:
        csv_data: CSV data with option parameters

    Returns:
        JSON with sensitivity validation results
    """
    try:
        df = _parse_csv_data(csv_data)

        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps({"status": "error", "message": f"Missing columns: {missing}"})

        if len(df) == 0:
            return json.dumps({"status": "error", "message": "No options in dataset"})

        first_row = df.iloc[0]
        option_type = str(first_row['option_type']).lower()
        S, K, T, r, sigma = (
            float(first_row['S']), float(first_row['K']),
            float(first_row['T']), float(first_row['r']),
            float(first_row['sigma'])
        )

        # Run sensitivity test (-2.5% to +2.5%, step 0.5%)
        spot_changes = np.arange(-0.025, 0.026, 0.005)
        sensitivity_data = []

        for change in spot_changes:
            new_S = S * (1 + float(change))
            greeks = _calculate_greeks(option_type, new_S, K, T, r, sigma)

            entry = {"spot_change": float(change)}
            if 'error' in greeks:
                entry['error'] = greeks['error']
            else:
                for k in ('price', 'delta', 'gamma', 'vega', 'rho', 'theta'):
                    entry[k] = greeks.get(k)

            sensitivity_data.append(entry)

        # Validate results
        results = {
            "option_tested": {"option_type": option_type, "S": S, "K": K, "T": T, "r": r, "sigma": sigma},
            "data_points": len(sensitivity_data),
            "validations": []
        }

        # Validation 1: Should have 11 entries
        expected_count = 11
        has_correct_count = len(sensitivity_data) == expected_count
        results['validations'].append({
            "rule": f"Has {expected_count} data points",
            "status": "passed" if has_correct_count else "failed",
            "value": len(sensitivity_data),
            "message": f"Found {len(sensitivity_data)} points" + ("" if has_correct_count else f", expected {expected_count}")
        })

        # Validation 2: All entries have required fields
        required_fields = ['spot_change', 'price', 'delta', 'gamma', 'vega', 'rho', 'theta']
        error_count = sum(1 for e in sensitivity_data if 'error' in e)
        missing_fields_count = sum(1 for e in sensitivity_data if 'error' not in e and any(f not in e for f in required_fields))

        all_valid = (error_count == 0 and missing_fields_count == 0)
        results['validations'].append({
            "rule": "All entries have required fields",
            "status": "passed" if all_valid else "failed",
            "message": "All valid" if all_valid else f"{missing_fields_count} missing fields, {error_count} errors"
        })

        results['sample_data'] = sensitivity_data[:3]
        results['status'] = 'passed' if all(v['status'] == 'passed' for v in results['validations']) else 'failed'

        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error: {str(e)}"})


def get_validator_tools():
    """Return list of validation tools."""
    return [
        batch_greeks_validator,
        validate_put_call_parity,
        validate_sensitivity
    ]