from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from src.agents.state import OptionAnalysisState
from src.agents.tools.calculator_tools import get_calculation_tools


import json

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
                    "calculator_agent_status": "failed",
                    "errors": ["No CSV data available in statue"]
                }

            # Create task message
            task_message = HumanMessage(content=f"""
Calculate Black-Scholes-Merton option prices and Greeks for the following data:

{csv_data}

Use batch_bsm_calculator for price calculation.

Return:
1. A json string with all option prices
""")

            # Invoke agent
            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract calculation results
            calculation_results = None
            greeks_data = None
            sensitivity_data = None

            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name'):
                    if msg.name == 'batch_bsm_calculator':
                        # batch_bsm_calculator 返回 markdown 字符串，直接使用
                        calculation_results = json.loads(msg.content)
                    # elif msg.name == 'greeks_calculator':  # 修复: 'w' → 'greeks_calculator'
                    #     # greeks_calculator 返回 JSON 字符串，需要解析
                    #     import json
                    #     try:
                    #         greeks_data = json.loads(msg.content)
                    #     except:
                    #         pass
                    # elif msg.name == 'sensitivity_test':  # 可选：也可以获取敏感性测试数据
                    #     import json
                    #     try:
                    #         sensitivity_data = json.loads(msg.content)
                    #     except:
                    #         pass

            return {
                "calculation_results": calculation_results or "No calculation results",
                # "greeks_data": greeks_data,
                # "sensitivity_data": sensitivity_data,
                "calculator_agent_status": "completed",
                "current_agent": "calculator",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "calculator_agent_status": "error",
                "current_agent": "calculator",
                "errors": [f"Error: {str(e)}"]
            }
