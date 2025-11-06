from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from src.agents.state import OptionAnalysisState
from src.agents.tools.summary_writer_tools import get_summary_writer_tools



class SummaryWriter:
    """Agent 4: Write textual summary"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_summary_writer_tools()
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Write summary based on all previous results"""
        try:
            calculation_results = state.get("calculation_results", "No results")
            test_results = state.get("test_results", {})
            csv_data = state.get("csv_data", {})

            task_message = HumanMessage(content=f"""
Generate a comprehensive summary report using the generate_summary tool.

Input data:
- Calculation Results: {calculation_results}
- Test Results: {test_results}
- CSV Data: {csv_data}

The tool will:
1. Load the template from src/templates/summary_template.md
2. Fill in all placeholders with actual data
3. Generate a professional markdown report
4. If template not found, create a structured summary automatically

Call the generate_summary tool with all three inputs.
""")

            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract summary from tool message
            summary_text = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name') and msg.name == 'generate_summary':
                    summary_text = msg.content
                    break

            return {
                "summary_text": summary_text or "Summary generation failed",
                "agent_status": "completed",
                "current_agent": "summary writer",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "agent_status": "error",
                "current_agent": "summary writer",
                "errors": [f"Error: {str(e)}"]
            }