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
Run comprehensive validation tests on ALL options loaded from CSV.

CSV Data: {csv_data}

Execute the following validation tests:

1. batch_greeks_validator - Validates Greeks for ALL {len(csv_data) if isinstance(csv_data, list) else 'loaded'} options:
   - For each option, calculates Greeks using its actual parameters
   - Validates: price > 0
   - Validates: delta in correct range (call: [0,1], put: [-1,0])
   - Validates: gamma >= 0, vega >= 0
   - Returns detailed pass/fail results for each option

2. validate_put_call_parity - Tests put-call parity for paired options:
   - Finds call/put pairs with matching parameters (S, K, T, r, sigma)
   - Validates: C - P ≈ S - K*e^(-rT)
   - Reports which pairs pass/fail the parity test

3. validate_sensitivity - Runs sensitivity analysis on first option:
   - Tests spot price sensitivity (-2.5% to +2.5%)
   - Validates 11 data points with all required Greeks fields
   - Ensures no calculation errors

Each tool accepts the csv_data as input and processes all relevant options.
Report the overall test status, number of options tested, and any validation failures found.
""")

            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract test results from tool messages
            test_results = {"tests_run": [], "overall_status": "unknown"}

            # Tool names to look for
            tool_names = ['batch_greeks_validator', 'validate_put_call_parity', 'validate_sensitivity']

            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name') and msg.name in tool_names:
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