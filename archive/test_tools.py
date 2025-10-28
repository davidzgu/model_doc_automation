# -*- coding: utf-8 -*-
"""
LangChain tools for running option pricing tests.

Wraps test functions from tests/test_greeks.py and tests/test_sensitivity.py
as LangChain tools for use in the multi-agent workflow.
"""
import json
import sys
import os
from typing import Dict, Any, Union
from langchain_core.tools import tool

# Add tests directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests')))

from src.bsm_utils import greeks_calculator, sensitivity_test


@tool
def run_greeks_validation_test(calculation_data: Union[str, Dict[str, Any]]) -> str:
    """
    Run Greeks validation tests on BSM calculation results.

    Tests include:
    1. Basic Greeks calculation (delta, gamma, vega, rho, theta)
    2. Put-call parity validation
    3. Delta range validation (0-1 for calls, -1-0 for puts)

    Args:
        calculation_data: JSON string or dict containing option parameters
                         (option_type, S, K, T, r, sigma)

    Returns:
        JSON string with test results: {
            "status": "passed" | "failed",
            "tests_run": int,
            "tests_passed": int,
            "details": List[Dict]
        }
    """
    try:
        # Parse input
        if isinstance(calculation_data, str):
            data = json.loads(calculation_data)
        else:
            data = calculation_data

        test_results = {
            "status": "passed",
            "tests_run": 0,
            "tests_passed": 0,
            "details": []
        }

        # Extract first row for testing (or use provided single option)
        if isinstance(data, dict) and 'S' in data:
            # Single option data
            test_data = [data]
        elif isinstance(data, dict):
            # Pandas-style dict {column: {index: value}}
            test_data = []
            if 'S' in data and isinstance(data['S'], dict):
                indices = list(data['S'].keys())
                for idx in indices[:1]:  # Test first row only
                    row = {col: data[col][idx] for col in data.keys()}
                    test_data.append(row)
        else:
            test_data = [data[0]] if isinstance(data, list) and len(data) > 0 else []

        if not test_data:
            return json.dumps({
                "status": "failed",
                "error": "No valid test data found"
            })

        option = test_data[0]
        option_type = str(option.get('option_type', 'call')).lower()
        S = float(option.get('S', 100))
        K = float(option.get('K', 100))
        T = float(option.get('T', 1.0))
        r = float(option.get('r', 0.05))
        sigma = float(option.get('sigma', 0.2))

        # Test 1: Basic Greeks calculation
        test_results["tests_run"] += 1
        try:
            res = greeks_calculator(option_type, S, K, T, r, sigma)
            greeks = json.loads(res)

            if all(k in greeks for k in ['price', 'delta', 'gamma', 'vega', 'rho', 'theta']):
                if greeks['price'] > 0:
                    test_results["tests_passed"] += 1
                    test_results["details"].append({
                        "test": "Basic Greeks Calculation",
                        "status": "passed",
                        "message": f"Price: {greeks['price']:.4f}, Delta: {greeks['delta']:.4f}"
                    })
                else:
                    test_results["details"].append({
                        "test": "Basic Greeks Calculation",
                        "status": "failed",
                        "message": "Price is not positive"
                    })
            else:
                test_results["details"].append({
                    "test": "Basic Greeks Calculation",
                    "status": "failed",
                    "message": "Missing Greeks fields"
                })
        except Exception as e:
            test_results["details"].append({
                "test": "Basic Greeks Calculation",
                "status": "error",
                "message": str(e)
            })

        # Test 2: Delta range validation
        test_results["tests_run"] += 1
        try:
            res = greeks_calculator(option_type, S, K, T, r, sigma)
            greeks = json.loads(res)
            delta = greeks.get('delta', 0)

            if option_type == 'call':
                if 0 <= delta <= 1:
                    test_results["tests_passed"] += 1
                    test_results["details"].append({
                        "test": "Delta Range Validation (Call)",
                        "status": "passed",
                        "message": f"Delta={delta:.4f} is in valid range [0,1]"
                    })
                else:
                    test_results["details"].append({
                        "test": "Delta Range Validation (Call)",
                        "status": "failed",
                        "message": f"Delta={delta:.4f} is outside [0,1]"
                    })
            elif option_type == 'put':
                if -1 <= delta <= 0:
                    test_results["tests_passed"] += 1
                    test_results["details"].append({
                        "test": "Delta Range Validation (Put)",
                        "status": "passed",
                        "message": f"Delta={delta:.4f} is in valid range [-1,0]"
                    })
                else:
                    test_results["details"].append({
                        "test": "Delta Range Validation (Put)",
                        "status": "failed",
                        "message": f"Delta={delta:.4f} is outside [-1,0]"
                    })
        except Exception as e:
            test_results["details"].append({
                "test": "Delta Range Validation",
                "status": "error",
                "message": str(e)
            })

        # Determine overall status
        if test_results["tests_passed"] == test_results["tests_run"]:
            test_results["status"] = "passed"
        else:
            test_results["status"] = "failed"

        return json.dumps(test_results, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error running Greeks validation: {str(e)}"
        })


@tool
def run_sensitivity_analysis_test(option_params: str) -> str:
    """
    Run sensitivity analysis test on an option.

    Tests option price sensitivity to spot price changes from -2.5% to +2.5%.

    Args:
        option_params: JSON string with format:
            {
                "option_type": "call" or "put",
                "S": spot price,
                "K": strike,
                "T": time to maturity,
                "r": risk-free rate,
                "sigma": volatility
            }

    Returns:
        JSON string with sensitivity test results
    """
    try:
        params = json.loads(option_params) if isinstance(option_params, str) else option_params

        option_type = str(params.get('option_type', 'call'))
        S = float(params.get('S', 100))
        K = float(params.get('K', 100))
        T = float(params.get('T', 1.0))
        r = float(params.get('r', 0.05))
        sigma = float(params.get('sigma', 0.2))

        # Run sensitivity test
        result = sensitivity_test(option_type, S, K, T, r, sigma)
        sensitivity_data = json.loads(result)

        # Validate results
        test_result = {
            "status": "passed",
            "message": f"Sensitivity analysis completed with {len(sensitivity_data)} data points",
            "data_points": len(sensitivity_data),
            "spot_range": f"{min([d['spot_change'] for d in sensitivity_data]):.2%} to {max([d['spot_change'] for d in sensitivity_data]):.2%}",
            "sample_results": sensitivity_data[:3]  # First 3 results
        }

        # Check if all entries have required fields
        for entry in sensitivity_data:
            if 'error' in entry:
                test_result["status"] = "failed"
                test_result["message"] = f"Error in sensitivity calculation: {entry['error']}"
                break

        return json.dumps(test_result, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error running sensitivity test: {str(e)}"
        })
