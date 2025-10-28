# -*- coding: utf-8 -*-
"""
Test for Agent 3: Test Agent

Tests that the test agent can successfully run validation tests.
"""
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.nodes.tester_agent import Tester
from src.config.llm_config import get_llm


def test_tester_agent():
    """
    Test the Test Agent (Agent 3).

    Validates:
    1. Agent can run validation tests
    2. Returns test_results with pass/fail status
    3. Executes actual pytest functions from test_greeks.py and test_sensitivity.py
    4. Sets tester_agent_status to 'completed'
    """
    print("\n" + "="*60)
    print("Testing Agent 3: Tester (Test Agent)")
    print("="*60)

    # Initialize LLM and agent
    llm = get_llm()
    agent = Tester(llm)

    # Create mock state with csv_data and calculation_results
    # (simulating Agent 1 and Agent 2 outputs)
    mock_csv_data = {
        "option_type": {0: "call", 1: "put"},
        "S": {0: 100.0, 1: 105.0},
        "K": {0: 100.0, 1: 100.0},
        "T": {0: 1.0, 1: 0.5},
        "r": {0: 0.05, 1: 0.05},
        "sigma": {0: 0.2, 1: 0.25}
    }

    mock_calculation_results = """
| option_type | S     | K     | price   | delta | gamma | vega  |
|-------------|-------|-------|---------|-------|-------|-------|
| call        | 100.0 | 100.0 | 10.45   | 0.637 | 0.019 | 39.70 |
| put         | 105.0 | 100.0 | 3.15    | -0.245| 0.021 | 28.43 |
"""

    initial_state = {
        "messages": [],
        "csv_file_path": "/inputs/dummy_options.csv",
        "csv_data": mock_csv_data,
        "data_loader_agent_status": "completed",
        "calculation_results": mock_calculation_results,
        "greeks_data": None,
        "sensitivity_data": None,
        "calculator_agent_status": "completed",
        "test_results": None,
        "tester_agent_status": None,
        "summary_text": None,
        "summarty_writer_agnet_status": None,
        "charts": None,
        "chart_descriptions": None,
        "chart_generator_agent_status": None,
        "final_report_path": None,
        "final_report_html": None,
        "report_assembler_agent_status": None,
        "current_agent": "agent2",
        "workflow_status": "in_progress",
        "errors": []
    }

    # Run agent
    print("\n" + "-"*60)
    print("Running Test Agent...")
    print("-"*60)
    print("\nThis agent will execute actual pytest test functions:")
    print("  - test_greeks_call_basic")
    print("  - test_greeks_put_call_parity")
    print("  - test_sensitivity_length_and_fields")

    try:
        result = agent(initial_state)

        print("\nAgent execution completed!")
        print(f"\nAgent Status: {result.get('tester_agent_status')}")
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
            assert result.get('tester_agent_status') == 'completed', \
                f"Expected tester_agent_status='completed', got '{result.get('tester_agent_status')}'"
            print("✓ Test 1 PASSED: tester_agent_status = 'completed'")
            assertions.append(("tester_agent_status check", True))
        except AssertionError as e:
            print(f"✗ Test 1 FAILED: {e}")
            assertions.append(("tester_agent_status check", False))

        # Test 2: test_results should not be None
        try:
            assert result.get('test_results') is not None, \
                "test_results should not be None"
            print("✓ Test 2 PASSED: test_results is not None")
            assertions.append(("test_results existence", True))
        except AssertionError as e:
            print(f"✗ Test 2 FAILED: {e}")
            assertions.append(("test_results existence", False))

        # Test 3: test_results should be a dict
        try:
            test_results = result.get('test_results')
            assert isinstance(test_results, dict), \
                f"test_results should be dict, got {type(test_results)}"
            print(f"✓ Test 3 PASSED: test_results is dict")
            assertions.append(("test_results type", True))
        except AssertionError as e:
            print(f"✗ Test 3 FAILED: {e}")
            assertions.append(("test_results type", False))

        # Test 4: test_results should have expected structure
        try:
            test_results = result.get('test_results', {})
            # Check for either 'tests_run' list or 'overall_status'
            has_structure = ('tests_run' in test_results) or ('overall_status' in test_results)
            assert has_structure, \
                f"test_results missing expected fields. Found: {list(test_results.keys())}"
            print(f"✓ Test 4 PASSED: test_results has expected structure")
            assertions.append(("test_results structure", True))
        except AssertionError as e:
            print(f"✗ Test 4 FAILED: {e}")
            assertions.append(("test_results structure", False))

        # Test 5: Check if tests were actually executed
        try:
            test_results = result.get('test_results', {})
            tests_run = test_results.get('tests_run', [])

            if isinstance(tests_run, list) and len(tests_run) > 0:
                # Look for test tool calls
                test_tools_called = [t.get('test_name', '') for t in tests_run]
                has_greeks_test = any('greeks' in name.lower() for name in test_tools_called)
                has_sensitivity_test = any('sensitivity' in name.lower() for name in test_tools_called)

                assert has_greeks_test or has_sensitivity_test, \
                    f"Expected at least one test to run. Tools called: {test_tools_called}"
                print(f"✓ Test 5 PASSED: Tests were executed ({len(tests_run)} tool calls)")
                assertions.append(("tests execution", True))
            else:
                # Check overall_status instead
                overall_status = test_results.get('overall_status', 'unknown')
                if overall_status != 'unknown':
                    print(f"✓ Test 5 PASSED: Overall status = '{overall_status}'")
                    assertions.append(("tests execution", True))
                else:
                    raise AssertionError("No evidence of test execution found")
        except AssertionError as e:
            print(f"✗ Test 5 FAILED: {e}")
            assertions.append(("tests execution", False))

        # Print test results details
        print("\n" + "-"*60)
        print("Test Results Details:")
        print("-"*60)
        test_results = result.get('test_results', {})
        print(json.dumps(test_results, indent=2, default=str))

        # Analyze test outcomes
        print("\n" + "-"*60)
        print("Test Outcomes Analysis:")
        print("-"*60)

        tests_run = test_results.get('tests_run', [])
        if tests_run:
            for test_info in tests_run:
                test_name = test_info.get('test_name', 'Unknown')
                test_result = test_info.get('result', {})

                print(f"\nTool: {test_name}")

                if isinstance(test_result, dict):
                    status = test_result.get('status', 'unknown')
                    tests_passed = test_result.get('tests_passed', 0)
                    total_tests = test_result.get('tests_run', 0)

                    print(f"  Status: {status}")
                    print(f"  Tests: {tests_passed}/{total_tests} passed")

                    # Show details
                    details = test_result.get('details', [])
                    if details:
                        print(f"  Details:")
                        for detail in details:
                            test_case = detail.get('test', 'Unknown test')
                            test_status = detail.get('status', 'unknown')
                            message = detail.get('message', '')
                            status_icon = "✓" if test_status == "passed" else "✗"
                            print(f"    {status_icon} {test_case}: {message}")
        else:
            print("No detailed test run information available")
            overall = test_results.get('overall_status', 'unknown')
            print(f"Overall status: {overall}")

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
    success = test_tester_agent()
    sys.exit(0 if success else 1)