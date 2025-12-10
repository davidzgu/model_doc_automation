from pathlib import Path
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role
from bsm_multi_agents.prompts.loader import load_prompt
from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.utils import get_tool_result_from_messages


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
    user_prompt = load_prompt(prompt_path)

    agent_input = state.copy()
    agent_input["messages"] = [HumanMessage(content=user_prompt)]
    agent_input["remaining_steps"] = 10
    result = agent.invoke(
        agent_input,
        config={"recursion_limit": 10, "configurable": {"thread_id": state.get("thread_id","run-1")}}
    )

    validate_results = get_tool_result_from_messages(result["messages"], "batch_greeks_validator")
    if "Error" in validate_results:
        return {
            "messages": result["messages"],
            "errors": validate_results["Error"]
        }
    else:
        return {
            "messages": result["messages"],
            "validate_results": json.loads(validate_results["result"])['validate_results'],
        }