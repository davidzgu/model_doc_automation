from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from src.agents.state import OptionAnalysisState
from src.agents.tools.chart_generator_tools import get_chart_generator_tools



class Agent5_ChartGenerator:
    """Agent 5: Generate charts and visualizations"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_chart_generator_tools()
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Generate charts"""
        try:
            csv_data = state.get("csv_data")
            greeks_data = state.get("greeks_data")

            task_message = HumanMessage(content=f"""
Generate visualization charts for the option analysis.

CSV Data: {csv_data}
Greeks Data: {greeks_data}

Use create_summary_charts tool to generate both price and Greeks charts.
Save them to outputs/charts/ directory.
""")

            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract chart paths
            charts = []
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name') and 'chart' in msg.name:
                    import json
                    try:
                        chart_info = json.loads(msg.content)
                        if chart_info.get("status") == "success":
                            if "charts" in chart_info:
                                charts.extend(chart_info["charts"])
                            elif "chart_path" in chart_info:
                                charts.append(chart_info["chart_path"])
                    except:
                        pass

            return {
                "charts": charts,
                "agent5_status": "completed",
                "current_agent": "agent5",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "agent5_status": "error",
                "current_agent": "agent5",
                "errors": [f"Agent 5 error: {str(e)}"]
            }
