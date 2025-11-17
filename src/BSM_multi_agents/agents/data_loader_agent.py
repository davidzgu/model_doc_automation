from pathlib import Path
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role
from bsm_multi_agents.prompts.loader import load_prompt
from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.utils import merge_state_update_from_tool_messages


def data_loader_node(
        state: WorkflowState,
) -> Dict[str, Any]:
    agent_role = 'data_loader'
    agent = built_graph_agent_by_role(agent_role)

    csv_path = state.get("csv_file_path")
    if not csv_path:
        csv_path = str(Path.cwd().parents[1] / "data" / "input" / "dummy_options.csv")

    prompt_path = Path.cwd().parents[1] / "src" / "bsm_multi_agents" / "prompts" / "data_loader_prompts.txt"
    prompt = load_prompt(prompt_path).format(csv_path=str(csv_path))
    msg = HumanMessage(content=prompt)

    result = agent.invoke(
        {"messages": [msg]},
        config={
            "recursion_limit": 10,
            "configurable": {"thread_id": "run-1"}
        }
    )

    merged_messages = list(state.get("messages", []))
    if isinstance(result, dict) and "messages" in result:
        merged_messages.extend(result["messages"])
    out = {"messages": merged_messages}

    merge_state_update_from_tool_messages(result, out, tool_names=("csv_loader",))

    return out






