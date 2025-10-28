# -*- coding: utf-8 -*-
"""
Test for Agent 2: Calculator Agent

Tests that the calculator agent can successfully calculate BSM prices and Greeks.
"""
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.nodes.calculator_agent import Calculator
from src.config.llm_config import get_llm


def test_calculator_agent():
    """
    Test the Calculator Agent (Agent 2).

    Validates:
    1. Agent can calculate BSM prices and Greeks
    2. Returns calculation_results in markdown table format
    3. Sets calculator_agent_status to 'completed'
    """
    print("\n" + "="*60)
    print("Testing Agent 2: Calculator")
    print("="*60)

    # Initialize LLM and agent
    llm = get_llm()
    agent = Calculator(llm)

    # Create mock state with csv_data (simulating Agent 1 output)
    mock_csv_data = {
        "option_type": {0: "call", 1: "put"},
        "S": {0: 100.0, 1: 105.0},
        "K": {0: 100.0, 1: 100.0},
        "T": {0: 1.0, 1: 0.5},
        "r": {0: 0.05, 1: 0.05},
        "sigma": {0: 0.2, 1: 0.25}
    }

    initial_state = {
        "messages": [],
        "csv_file_path": "/inputs/dummy_options.csv",
        "csv_data": mock_csv_data,
        "data_loader_agent_status": "completed",
        "calculation_results": None,
        "greeks_data": None,
        "sensitivity_data": None,
        "calculator_agent_status": None,
        "test_results": None,
        "test_agent_status": None,
        "summary_text": None,
        "summarty_writer_agnet_status": None,
        "charts": None,
        "chart_descriptions": None,
        "chart_generator_agent_status": None,
        "final_report_path": None,
        "final_report_html": None,
        "report_assembler_agent_status": None,
        "current_agent": "calculator",
        "workflow_status": "in_progress",
        "errors": []
    }

    # Run agent
    print("\n" + "-"*60)
    print("Running Calculator Agent...")
    print("-"*60)
    print(f"\nInput CSV Data:")
    print(json.dumps(mock_csv_data, indent=2))

    try:
        result = agent(initial_state)

        print("\nAgent execution completed!")
        print(f"\nAgent Status: {result.get('calculator_agent_status')}")
        print(f"Current Agent: {result.get('current_agent')}")
        print(f"Workflow Status: {result.get('workflow_status')}")

        # Check for errors
        if result.get('errors'):
            print(f"\nErrors: {result.get('errors')}")

        # Validate results
        print("\n" + "-"*60)
        print("Validation Results:")
        print("-"*60)

        assertions = []

        # Test 1: Agent status should be 'completed'
        try:
            assert result.get('calculator_agent_status') == 'completed', \
                f"Expected calculator_agent_status='completed', got '{result.get('calculator_agent_status')}'"
            print("✓ Test 1 PASSED: calculator_agent_status = 'completed'")
            assertions.append(("calculator_agent_status check", True))
        except AssertionError as e:
            print(f"✗ Test 1 FAILED: {e}")
            assertions.append(("calculator_agent_status check", False))

        # Test 2: calculation_results should not be None
        try:
            assert result.get('calculation_results') is not None, \
                "calculation_results should not be None"
            print("✓ Test 2 PASSED: calculation_results is not None")
            assertions.append(("calculation_results existence", True))
        except AssertionError as e:
            print(f"✗ Test 2 FAILED: {e}")
            assertions.append(("calculation_results existence", False))

        # Test 3: calculation_results should be a string (markdown table)
        try:
            calc_results = result.get('calculation_results')
            assert isinstance(calc_results, str), \
                f"calculation_results should be string, got {type(calc_results)}"
            print(f"✓ Test 3 PASSED: calculation_results is string")
            assertions.append(("calculation_results type", True))
        except AssertionError as e:
            print(f"✗ Test 3 FAILED: {e}")
            assertions.append(("calculation_results type", False))

        # Test 4: calculation_results should contain expected columns
        try:
            calc_results = result.get('calculation_results', '')
            # Check for markdown table headers
            expected_columns = ['option_type', 'price', 'delta', 'gamma', 'vega']
            has_columns = any(col in calc_results.lower() for col in expected_columns)
            assert has_columns, \
                f"calculation_results should contain pricing/Greeks columns"
            print(f"✓ Test 4 PASSED: calculation_results contains expected columns")
            assertions.append(("calculation_results content", True))
        except AssertionError as e:
            print(f"✗ Test 4 FAILED: {e}")
            assertions.append(("calculation_results content", False))

        # Test 5: Check if greeks_data exists (optional but expected)
        try:
            greeks_data = result.get('greeks_data')
            if greeks_data is not None:
                assert isinstance(greeks_data, (dict, list)), \
                    f"greeks_data should be dict or list, got {type(greeks_data)}"
                print(f"✓ Test 5 PASSED: greeks_data exists and is {type(greeks_data).__name__}")
                assertions.append(("greeks_data", True))
            else:
                print("⚠ Test 5 SKIPPED: greeks_data is None (optional)")
                assertions.append(("greeks_data", True))  # Not a failure
        except AssertionError as e:
            print(f"✗ Test 5 FAILED: {e}")
            assertions.append(("greeks_data", False))

        # Print calculation results
        print("\n" + "-"*60)
        print("Calculation Results:")
        print("-"*60)
        calc_results = result.get('calculation_results', 'No results')
        # Truncate if too long
        if len(calc_results) > 500:
            print(calc_results[:500] + "\n... (truncated)")
        else:
            print(calc_results)

        # Print Greeks data if available
        if result.get('greeks_data'):
            print("\n" + "-"*60)
            print("Greeks Data:")
            print("-"*60)
            print(json.dumps(result.get('greeks_data'), indent=2, default=str))

        # Final summary
        print("\n" + "="*60)
        print("Test Summary:")
        print("="*60)
        passed = sum(1 for _, status in assertions if status)
        total = len(assertions)
        print(f"Passed: {passed}/{total}")

        for test_name, status in assertions:
            status_str = "✓ PASS" if status else "✗ FAIL"
            print(f"  {status_str}: {test_name}")

        # Overall result
        if passed == total:
            print("\n🎉 All tests PASSED!")
            return True
        else:
            print(f"\n⚠️  {total - passed} test(s) FAILED")
            return False

    except Exception as e:
        print(f"\n❌ Test execution error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_calculator_agent()
    sys.exit(0 if success else 1)