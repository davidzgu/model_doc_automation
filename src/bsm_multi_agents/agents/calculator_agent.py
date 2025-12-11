from pathlib import Path
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role
from bsm_multi_agents.prompts.loader import load_prompt
from bsm_multi_agents.graph.state import WorkflowState
#from bsm_multi_agents.agents.utils import merge_state_update_from_tool_messages


def calculator_node(
        state: WorkflowState,
) -> Dict[str, Any]:
    if "csv_data" not in state or not state["csv_data"]:
        return {
            "messages": state.get("messages", []),
            "errors": ["calculator_node: csv_data is missing"],
        }
    
    agent_role = "calculator"
    system_prompt = """
    You are a quantitative calculator agent.
    """
    agent = built_graph_agent_by_role(agent_role,system_prompt=system_prompt)

    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "calculator_prompts.txt"
    print(prompt_path)
    csv_json = json.dumps(state["csv_data"], ensure_ascii=False)
    print(csv_json)
    user_prompt = load_prompt(prompt_path).format(csv_data_json=csv_json)

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_prompt)]},
        config={"recursion_limit": 10, "configurable": {"thread_id": state.get("thread_id","run-1")}}
    )

    merged_messages = list(state.get("messages", []))
    if isinstance(result, dict) and "messages" in result:
        merged_messages.extend(result["messages"])
    out = {"messages": merged_messages}

    merge_state_update_from_tool_messages(
        result,
        out,
        tool_names=("batch_bsm_calculator", "batch_greeks_calculator"),
    )


    return out

