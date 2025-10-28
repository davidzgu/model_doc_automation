# -*- coding: utf-8 -*-
"""
Test the new greeks_validator functionality.

Tests the three validation tools:
1. batch_greeks_validator
2. validate_put_call_parity
3. validate_sensitivity_test
"""
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.bsm_validator import (
    batch_greeks_validator,
    validate_put_call_parity,
    validate_sensitivity
)


def test_batch_greeks_validator():
    """Test batch Greeks validation with sample data."""
    print("\n" + "="*70)
    print("TEST 1: Batch Greeks Validator")
    print("="*70)

    # Sample CSV data with mixed call and put options
    sample_data = {
        "option_type": {0: "call", 1: "put", 2: "call"},
        "S": {0: 100.0, 1: 100.0, 2: 105.0},
        "K": {0: 100.0, 1: 100.0, 2: 100.0},
        "T": {0: 1.0, 1: 1.0, 2: 0.5},
        "r": {0: 0.05, 1: 0.05, 2: 0.05},
        "sigma": {0: 0.2, 1: 0.2, 2: 0.25}
    }

    print("\nSample Data:")
    print(json.dumps(sample_data, indent=2))

    # Run validation
    result_json = batch_greeks_validator.invoke({"csv_data": sample_data})
    result = json.loads(result_json)

    print("\n" + "-"*70)
    print("Validation Results:")
    print("-"*70)
    print(f"Status: {result['status']}")
    print(f"Total Options: {result['total_options']}")
    print(f"Passed: {result['passed']}")
    print(f"Failed: {result['failed']}")
    print(f"Errors: {result['errors']}")

    # Show details for each option
    for detail in result['details']:
        idx = detail['option_index']
        params = detail['parameters']
        status = detail['overall_status']

        print(f"\nOption {idx}: {params['option_type'].upper()} (S={params['S']}, K={params['K']})")
        print(f"  Overall Status: {status}")

        if 'greeks' in detail:
            greeks = detail['greeks']
            print(f"  Price: {greeks.get('price', 'N/A'):.4f}")
            print(f"  Delta: {greeks.get('delta', 'N/A'):.4f}")
            print(f"  Gamma: {greeks.get('gamma', 'N/A'):.4f}")
            print(f"  Vega:  {greeks.get('vega', 'N/A'):.4f}")

        if 'validations' in detail:
            for validation in detail['validations']:
                status_icon = "✓" if validation['status'] == 'passed' else "✗"
                print(f"  {status_icon} {validation['rule']}: {validation['message']}")

    return result['status'] == 'passed'


def test_validate_put_call_parity():
    """Test put-call parity validation."""
    print("\n" + "="*70)
    print("TEST 2: Put-Call Parity Validator")
    print("="*70)

    # Sample data with matching call/put pairs
    sample_data = {
        "option_type": {0: "call", 1: "put", 2: "call", 3: "put"},
        "S": {0: 100.0, 1: 100.0, 2: 105.0, 3: 105.0},
        "K": {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0},
        "T": {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5},
        "r": {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05},
        "sigma": {0: 0.2, 1: 0.2, 2: 0.25, 3: 0.25}
    }

    print("\nSample Data (2 call/put pairs with matching parameters):")
    print(json.dumps(sample_data, indent=2))

    # Run validation
    result_json = validate_put_call_parity.invoke({"csv_data": sample_data})
    result = json.loads(result_json)

    print("\n" + "-"*70)
    print("Parity Test Results:")
    print("-"*70)
    print(f"Status: {result['status']}")
    print(f"Pairs Found: {result['pairs_found']}")
    print(f"Pairs Tested: {result['pairs_tested']}")
    print(f"Pairs Passed: {result['pairs_passed']}")
    print(f"Pairs Failed: {result['pairs_failed']}")

    # Show details for each pair
    for detail in result.get('details', []):
        print(f"\nPair: Call[{detail['call_index']}] & Put[{detail['put_index']}]")
        print(f"  Parameters: S={detail['parameters']['S']}, K={detail['parameters']['K']}, T={detail['parameters']['T']}")
        print(f"  Call Price: {detail.get('call_price', 'N/A'):.4f}")
        print(f"  Put Price:  {detail.get('put_price', 'N/A'):.4f}")
        print(f"  C - P = {detail.get('lhs', 'N/A'):.4f}")
        print(f"  S - K*e^(-rT) = {detail.get('rhs', 'N/A'):.4f}")
        print(f"  Difference: {detail.get('difference', 'N/A'):.6f}")
        print(f"  Status: {detail['status']} - {detail.get('message', '')}")

    return result['status'] in ['passed', 'partial']


def test_validate_sensitivity():
    """Test sensitivity analysis validation."""
    print("\n" + "="*70)
    print("TEST 3: Sensitivity Validator")
    print("="*70)

    # Sample data (will test first option)
    sample_data = {
        "option_type": {0: "call"},
        "S": {0: 100.0},
        "K": {0: 100.0},
        "T": {0: 1.0},
        "r": {0: 0.05},
        "sigma": {0: 0.2}
    }

    print("\nSample Data (testing first option):")
    print(json.dumps(sample_data, indent=2))

    # Run validation
    result_json = validate_sensitivity.invoke({"csv_data": sample_data})
    result = json.loads(result_json)

    print("\n" + "-"*70)
    print("Sensitivity Test Results:")
    print("-"*70)
    print(f"Status: {result['status']}")

    if 'option_tested' in result:
        opt = result['option_tested']
        print(f"Option Tested: {opt['option_type'].upper()} (S={opt['S']}, K={opt['K']}, T={opt['T']})")

    print(f"Data Points: {result.get('data_points', 'N/A')}")

    # Show validation results
    for validation in result.get('validations', []):
        status_icon = "✓" if validation['status'] == 'passed' else "✗"
        print(f"  {status_icon} {validation['rule']}: {validation['message']}")

    # Show sample data
    if 'sample_data' in result:
        print("\nSample Sensitivity Data (first 3 points):")
        print(json.dumps(result['sample_data'], indent=2))

    return result['status'] == 'passed'


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("GREEKS VALIDATOR FUNCTIONALITY TESTS")
    print("="*70)

    results = {
        "batch_greeks_validator": False,
        "put_call_parity": False,
        "sensitivity_test": False
    }

    try:
        results["batch_greeks_validator"] = test_batch_greeks_validator()
    except Exception as e:
        print(f"\n❌ Test 1 Error: {str(e)}")
        import traceback
        traceback.print_exc()

    try:
        results["put_call_parity"] = test_validate_put_call_parity()
    except Exception as e:
        print(f"\n❌ Test 2 Error: {str(e)}")
        import traceback
        traceback.print_exc()

    try:
        results["sensitivity_test"] = test_validate_sensitivity()
    except Exception as e:
        print(f"\n❌ Test 3 Error: {str(e)}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All validator tests PASSED!")
    else:
        failed_count = sum(1 for p in results.values() if not p)
        print(f"\n⚠️  {failed_count} test(s) FAILED")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)