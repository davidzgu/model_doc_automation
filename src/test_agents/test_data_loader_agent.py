# -*- coding: utf-8 -*-
"""
Test for Agent 1: Data Loader Agent

Tests that the data loader agent can successfully load CSV data.
"""
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.nodes.data_loader_agent import DataLoader
from src.config.llm_config import get_llm


def test_data_loader_agent():
    """
    Test the Data Loader Agent (Agent 1).

    Validates:
    1. Agent can load CSV file
    2. Returns csv_data in correct format
    3. Sets agent1_status to 'completed'
    """
    print("\n" + "="*60)
    print("Testing Agent 1: Data Loader")
    print("="*60)

    # Initialize LLM and agent
    llm = get_llm()
    agent = DataLoader(llm)

    # Create initial state
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'inputs', 'dummy_options.csv')
    csv_path = os.path.abspath(csv_path)

    print(f"\nTest CSV path: {csv_path}")
    print(f"File exists: {os.path.exists(csv_path)}")

    initial_state = {
        "messages": [],
        "csv_file_path": csv_path,
        "csv_data": None,
        "data_loader_agent_status": None,
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
        "current_agent": None,
        "workflow_status": "started",
        "errors": []
    }

    # Run agent
    print("\n" + "-"*60)
    print("Running Data Loader Agent...")
    print("-"*60)

    try:
        result = agent(initial_state)

        print("\nAgent execution completed!")
        print(f"\nAgent Status: {result.get('agent1_status')}")
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
            assert result.get('agent_status') == 'completed', \
                f"Expected agent_status='completed', got '{result.get('agent_status')}'"
            print("✓ Test 1 PASSED: agent_status = 'completed'")
            assertions.append(("agent_status check", True))
        except AssertionError as e:
            print(f"✗ Test 1 FAILED: {e}")
            assertions.append(("agent_status check", False))

        # Test 2: csv_data should not be None
        try:
            assert result.get('csv_data') is not None, "csv_data should not be None"
            print("✓ Test 2 PASSED: csv_data is not None")
            assertions.append(("csv_data existence", True))
        except AssertionError as e:
            print(f"✗ Test 2 FAILED: {e}")
            assertions.append(("csv_data existence", False))

        # Test 3: csv_data should be a dict or list
        try:
            csv_data = result.get('csv_data')
            assert isinstance(csv_data, (dict, list)), \
                f"csv_data should be dict or list, got {type(csv_data)}"
            print(f"✓ Test 3 PASSED: csv_data is {type(csv_data).__name__}")
            assertions.append(("csv_data type", True))
        except AssertionError as e:
            print(f"✗ Test 3 FAILED: {e}")
            assertions.append(("csv_data type", False))

        # Test 4: csv_data should contain expected fields
        try:
            csv_data = result.get('csv_data')
            if isinstance(csv_data, dict):
                # Check if it's pandas-style dict with column names
                expected_fields = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
                has_fields = all(field in csv_data for field in expected_fields)
                assert has_fields, f"csv_data missing expected fields. Found: {list(csv_data.keys())}"
                print(f"✓ Test 4 PASSED: csv_data has all expected fields")
                assertions.append(("csv_data fields", True))
            elif isinstance(csv_data, list) and len(csv_data) > 0:
                # Check if it's list of dicts
                first_row = csv_data[0]
                expected_fields = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
                has_fields = all(field in first_row for field in expected_fields)
                assert has_fields, f"csv_data rows missing expected fields. Found: {list(first_row.keys())}"
                print(f"✓ Test 4 PASSED: csv_data rows have all expected fields")
                assertions.append(("csv_data fields", True))
            else:
                raise AssertionError("csv_data format not recognized")
        except AssertionError as e:
            print(f"✗ Test 4 FAILED: {e}")
            assertions.append(("csv_data fields", False))

        # Print CSV data preview
        print("\n" + "-"*60)
        print("CSV Data Preview:")
        print("-"*60)
        csv_data = result.get('csv_data')
        if isinstance(csv_data, dict):
            # Pandas-style dict
            print(json.dumps({k: v for k, v in list(csv_data.items())[:3]}, indent=2, default=str))
        elif isinstance(csv_data, list):
            # List of dicts
            print(json.dumps(csv_data[:2], indent=2, default=str))

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
    success = test_data_loader_agent()
    sys.exit(0 if success else 1)