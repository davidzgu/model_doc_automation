from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from src.agents.state import OptionAnalysisState
from src.agents.tools.report_assemble_tools import get_report_assembler_tools



class Agent6_ReportAssembler:
    """Agent 6: Assemble final report"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_report_assembler_tools()
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Assemble final HTML report"""
        try:
            summary_text = state.get("summary_text", "No summary")
            charts = state.get("charts", [])

            task_message = HumanMessage(content=f"""
Assemble the final HTML report combining the summary and charts.

Summary:
{summary_text}

Charts: {charts}

Use assemble_html_report tool to create a complete HTML report.
Save it to outputs/report.html
""")

            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract report path
            report_path = None
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name') and msg.name == 'assemble_html_report':
                    import json
                    try:
                        report_info = json.loads(msg.content)
                        if report_info.get("status") == "success":
                            report_path = report_info.get("report_path")
                    except:
                        pass

            return {
                "final_report_path": report_path or "Report generation failed",
                "agent6_status": "completed",
                "current_agent": "agent6",
                "workflow_status": "completed"
            }

        except Exception as e:
            return {
                "agent6_status": "error",
                "current_agent": "agent6",
                "workflow_status": "failed",
                "errors": [f"Agent 6 error: {str(e)}"]
            }
