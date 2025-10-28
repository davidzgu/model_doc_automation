from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from src.agents.state import OptionAnalysisState
from src.agents.tools.tester_tools import get_tester_tools



class Tester:
    """Agent 3: Run validation tests"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_tester_tools()
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Run tests on calculation results"""
        try:
            csv_data = state.get("csv_data")
            if not csv_data:
                return {
                    "tester_agent_status": "skipped",
                    "test_results": {"status": "skipped", "message": "No data to test"}
                }

            task_message = HumanMessage(content=f"""
Run validation tests on the option calculations.

Execute the following tests from src/core:
1. run_greeks_validation_test - runs actual pytest functions:
   - test_greeks_call_basic: validates basic Greeks calculation (delta in [0,1], price > 0)
   - test_greeks_put_call_parity: validates put-call parity relationship

2. run_sensitivity_analysis_test - runs actual pytest function:
   - test_sensitivity_length_and_fields: validates sensitivity analysis returns 11 data points with all required fields

These tools wrap the actual test functions from test_greeks.py and test_sensitivity.py.

Report the test results including pass/fail status for each test.
""")

            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract test results
            test_results = {"tests_run": [], "overall_status": "unknown"}

            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name') and 'test' in msg.name:
                    import json
                    try:
                        test_data = json.loads(msg.content)
                        test_results["tests_run"].append({
                            "test_name": msg.name,
                            "result": test_data
                        })
                    except:
                        pass

            if test_results["tests_run"]:
                test_results["overall_status"] = "completed"

            return {
                "test_results": test_results,
                "tester_agent_status": "completed",
                "current_agent": "tester",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "tester_agent_status": "error",
                "current_agent": "tester",
                "errors": [f"Error: {str(e)}"]
            }