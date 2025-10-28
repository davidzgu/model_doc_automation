from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from src.agents.state import OptionAnalysisState
from src.agents.tools.data_loader_tools import get_data_loader_tools

class DataLoader:
    """Agent 1: Load CSV data containing option parameters"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_data_loader_tools()
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Load CSV data"""
        try:
            csv_path = state["csv_file_path"]

            # Create task message
            task_message = HumanMessage(content=f"""
Load the option data from the CSV file at: {csv_path}

Use the csv_loader tool to read the CSV file. Return the data in JSON format.
""")

            # Invoke agent
            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract output from last message
            if result.get("messages"):
                # Look for tool messages with CSV data
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'name') and msg.name == 'csv_loader':
                        import json
                        try:
                            csv_data = json.loads(msg.content)
                            return {
                                "csv_data": csv_data,
                                "agent_status": "completed",
                                "current_agent": "data_loader",
                                "workflow_status": "in_progress"
                            }
                        except:
                            pass

            return {
                "agent_status": "failed",
                "current_agent": "data_loader",
                "errors": ["Failed to load CSV data"]
            }

        except Exception as e:
            return {
                "agent_status": "error",
                "current_agent": "data_loader",
                "errors": [f"Error: {str(e)}"]
            }