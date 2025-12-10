from pathlib import Path
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role
from bsm_multi_agents.prompts.loader import load_prompt
from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.utils import merge_state_update_from_tool_messages


def validator_node(
        state: WorkflowState,
) -> Dict[str, Any]:
    if "greeks_results" not in state or not state["greeks_results"]:
        return {
            "messages": state.get("messages", []),
            "errors": ["validator_node: greeks_results is missing"],
        }
    
    agent_role = "validator"
    default_system = """
    You are a quantitative validator agent.
    """
    agent = built_graph_agent_by_role(agent_role,default_system=default_system)

    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "validator_prompts.txt"
    greeks_results_json = json.dumps(state["greeks_results"], ensure_ascii=False)
    user_prompt = load_prompt(prompt_path).format(greeks_results=greeks_results_json)

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
        tool_names=("batch_greeks_validator",),
    )

    return out