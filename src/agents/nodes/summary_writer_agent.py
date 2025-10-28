from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from src.agents.state import OptionAnalysisState
from src.agents.tools.summary_writer_tools import get_summary_writer_tools



class Agent4_SummaryWriter:
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

            task_message = HumanMessage(content=f"""
Write a comprehensive summary of the option analysis.

Calculation Results:
{calculation_results}

Test Results:
{test_results}

Create a professional markdown summary with the following sections:
1. Overview
2. Key Findings
3. Calculation Summary
4. Test Results Summary
5. Conclusions

Keep it concise and clear.
""")

            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Get summary from last AI message
            summary_text = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'content') and len(msg.content) > 100:
                    summary_text = msg.content
                    break

            return {
                "summary_text": summary_text or "Summary generation failed",
                "agent4_status": "completed",
                "current_agent": "agent4",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "agent4_status": "error",
                "current_agent": "agent4",
                "errors": [f"Agent 4 error: {str(e)}"]
            }