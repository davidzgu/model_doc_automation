from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from src.agents.state import OptionAnalysisState
from src.agents.tools.test_tools import get_test_tools



class Tester:
    """Agent 3: Run validation tests"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_test_tools()
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Run tests on calculation results"""
        try:
            csv_data = state.get("csv_data")
            if not csv_data:
                return {
                    "agent3_status": "skipped",
                    "test_results": {"status": "skipped", "message": "No data to test"}
                }

            task_message = HumanMessage(content=f"""
Run validation tests on the option calculations.

Data: {csv_data}

Use the following tools:
1. run_greeks_validation_test - to validate Greeks calculations
2. run_sensitivity_analysis_test - to test sensitivity analysis

Report the test results.
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
                "agent3_status": "completed",
                "current_agent": "agent3",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "agent3_status": "error",
                "current_agent": "agent3",
                "errors": [f"Agent 3 error: {str(e)}"]
            }