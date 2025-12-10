from pathlib import Path
from typing import Dict, Any
import json

from langchain_core.messages import HumanMessage

from bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role
from bsm_multi_agents.prompts.loader import load_prompt
from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.utils import get_tool_result_from_messages,print_resp


def data_loader_node(
        state: WorkflowState,
) -> WorkflowState:
    agent_role = 'data_loader'
    agent = built_graph_agent_by_role(agent_role)

    csv_path = state.get("csv_file_path")
    if not csv_path:
        return {"messages": "No csv file path provided"}

    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "data_loader_prompts.txt"
    prompt = load_prompt(prompt_path).format(csv_path=str(csv_path))
    msg = HumanMessage(content=prompt)

    agent_input = {
        "messages": [msg],
        "remaining_steps": 10
    }
    result = agent.invoke(
        agent_input,
        config={
            "recursion_limit": 10,
            "configurable": {"thread_id": "run-1"}
        }
    )

    csv_data = get_tool_result_from_messages(result["messages"], "csv_loader")
    if "Error" in csv_data:
        return {
            "messages": result["messages"],
            "errors": csv_data["Error"]
        }
    else:
        return {
            "messages": result["messages"],
            "csv_data": json.loads(csv_data["result"])['csv_data']
        }



########


def main():
    csv_path = str(Path(__file__).resolve().parents[3] / "data" / "input" / "dummy_options.csv")
    state = WorkflowState(csv_file_path=csv_path)
    out = data_loader_node(state)
    print_resp(out)

if __name__ == "__main__":
    main()




