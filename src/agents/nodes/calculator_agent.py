from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from src.agents.state import OptionAnalysisState
from src.agents.tools.calculator_tools import get_calculation_tools




class Calculator:
    """Agent 2: Calculate BSM prices and Greeks"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_calculation_tools()
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Calculate option prices and Greeks"""
        try:
            csv_data = state.get("csv_data")
            if not csv_data:
                return {
                    "agent2_status": "failed",
                    "errors": ["No CSV data available from Agent 1"]
                }

            # Create task message
            task_message = HumanMessage(content=f"""
Calculate Black-Scholes-Merton option prices and Greeks for the following data:

{csv_data}

Use batch_bsm_calculator for price calculation and sensitivity_test for one sample option to get Greeks data.

Return:
1. A markdown table with all option prices
2. Greeks sensitivity analysis for the first option
""")

            # Invoke agent
            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract calculation results
            calculation_results = None
            greeks_data = None

            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name'):
                    if msg.name == 'batch_bsm_calculator':
                        calculation_results = msg.content
                    elif msg.name == 'w':
                        import json
                        try:
                            greeks_data = json.loads(msg.content)
                        except:
                            pass

            return {
                "calculation_results": calculation_results or "No calculation results",
                "greeks_data": greeks_data,
                "agent2_status": "completed",
                "current_agent": "agent2",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "agent2_status": "error",
                "current_agent": "agent2",
                "errors": [f"Agent 2 error: {str(e)}"]
            }
