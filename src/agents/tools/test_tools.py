# -*- coding: utf-8 -*-
"""
LangChain tools for running option pricing tests.

Wraps test functions from src/core/test_greeks.py and src/core/test_sensitivity.py
as LangChain tools for use in the multi-agent workflow.
"""
import json
import sys
import os
from typing import Dict, Any
from langchain_core.tools import tool

# Add core directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'core')))

# Import the actual pytest test functions
from test_greeks import test_greeks_call_basic, test_greeks_put_call_parity
from test_sensitivity import test_sensitivity_length_and_fields


@tool
def run_greeks_validation_test() -> str:
    """
    Run Greeks validation tests from src/core/test_greeks.py.

    Executes the following pytest test functions:
    1. test_greeks_call_basic - validates basic Greeks calculation (delta, gamma, vega, rho, theta)
    2. test_greeks_put_call_parity - validates put-call parity relationship

    Returns:
        JSON string with test results: {
            "status": "passed" | "failed",
            "tests_run": int,
            "tests_passed": int,
            "details": List[Dict]
        }
    """
    test_results = {
        "status": "passed",
        "tests_run": 0,
        "tests_passed": 0,
        "details": []
    }

    # Test 1: test_greeks_call_basic
    test_results["tests_run"] += 1
    try:
        test_greeks_call_basic()
        test_results["tests_passed"] += 1
        test_results["details"].append({
            "test": "test_greeks_call_basic",
            "status": "passed",
            "message": "Basic Greeks calculation validation passed"
        })
    except AssertionError as e:
        test_results["details"].append({
            "test": "test_greeks_call_basic",
            "status": "failed",
            "message": f"Assertion failed: {str(e)}"
        })
    except Exception as e:
        test_results["details"].append({
            "test": "test_greeks_call_basic",
            "status": "error",
            "message": f"Error: {str(e)}"
        })

    # Test 2: test_greeks_put_call_parity
    test_results["tests_run"] += 1
    try:
        test_greeks_put_call_parity()
        test_results["tests_passed"] += 1
        test_results["details"].append({
            "test": "test_greeks_put_call_parity",
            "status": "passed",
            "message": "Put-call parity validation passed"
        })
    except AssertionError as e:
        test_results["details"].append({
            "test": "test_greeks_put_call_parity",
            "status": "failed",
            "message": f"Assertion failed: {str(e)}"
        })
    except Exception as e:
        test_results["details"].append({
            "test": "test_greeks_put_call_parity",
            "status": "error",
            "message": f"Error: {str(e)}"
        })

    # Determine overall status
    if test_results["tests_passed"] == test_results["tests_run"]:
        test_results["status"] = "passed"
    else:
        test_results["status"] = "failed"

    return json.dumps(test_results, indent=2)


@tool
def run_sensitivity_analysis_test() -> str:
    """
    Run sensitivity analysis test from src/core/test_sensitivity.py.

    Executes test_sensitivity_length_and_fields which:
    - Tests option price sensitivity to spot price changes from -2.5% to +2.5%
    - Validates returned structure has 11 entries
    - Verifies all entries have required fields (spot_change, price, Greeks)

    Returns:
        JSON string with sensitivity test results: {
            "status": "passed" | "failed",
            "tests_run": int,
            "tests_passed": int,
            "details": List[Dict]
        }
    """
    test_results = {
        "status": "passed",
        "tests_run": 0,
        "tests_passed": 0,
        "details": []
    }

    # Test: test_sensitivity_length_and_fields
    test_results["tests_run"] += 1
    try:
        test_sensitivity_length_and_fields()
        test_results["tests_passed"] += 1
        test_results["details"].append({
            "test": "test_sensitivity_length_and_fields",
            "status": "passed",
            "message": "Sensitivity analysis validation passed (11 entries with all required fields)"
        })
    except AssertionError as e:
        test_results["details"].append({
            "test": "test_sensitivity_length_and_fields",
            "status": "failed",
            "message": f"Assertion failed: {str(e)}"
        })
    except Exception as e:
        test_results["details"].append({
            "test": "test_sensitivity_length_and_fields",
            "status": "error",
            "message": f"Error: {str(e)}"
        })

    # Determine overall status
    if test_results["tests_passed"] == test_results["tests_run"]:
        test_results["status"] = "passed"
    else:
        test_results["status"] = "failed"

    return json.dumps(test_results, indent=2)


def get_test_tools():
    """Return list of test tools for the Test Agent."""
    return [
        run_greeks_validation_test,
        run_sensitivity_analysis_test
    ]