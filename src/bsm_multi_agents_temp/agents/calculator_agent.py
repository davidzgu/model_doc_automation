from pathlib import Path
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role
from bsm_multi_agents.prompts.loader import load_prompt
from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.utils import get_tool_result_from_messages


def calculator_node(
        state: WorkflowState,
) -> Dict[str, Any]:
    if "csv_data" not in state or not state["csv_data"]:
        return {
            "messages": state.get("messages", []),
            "errors": ["calculator_node: csv_data is missing"],
        }
    
    agent_role = "calculator"
    default_system = """
    You are a quantitative calculator agent.
    """
    agent = built_graph_agent_by_role(agent_role,default_system=default_system)

    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "calculator_prompts.txt"
    user_prompt = load_prompt(prompt_path)

    agent_input = state.copy()
    agent_input["messages"] = [HumanMessage(content=user_prompt)]
    agent_input["remaining_steps"] = 10

    result = agent.invoke(
        agent_input,
        config={"recursion_limit": 10, "configurable": {"thread_id": state.get("thread_id","run-1")}}
    )

    bsm_result = get_tool_result_from_messages(result["messages"], "batch_bsm_calculator")
    greeks_result = get_tool_result_from_messages(result["messages"], "batch_greeks_calculator")
    if "Error" in bsm_result:
        return {
            "messages": result["messages"],
            "errors": bsm_result["Error"]
        }
    elif "Error" in greeks_result:
        return {
            "messages": result["messages"],
            "errors": greeks_result["Error"]
        }
    else:
        return {
            "messages": result["messages"],
            "bsm_results": json.loads(bsm_result["result"])['bsm_results'],
            "greeks_results": json.loads(greeks_result["result"])['greeks_results']
        }
    
