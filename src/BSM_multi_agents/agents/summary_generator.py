from pathlib import Path
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from bsm_multi_agents.agents.agent_factory import built_graph_agent_by_role
from bsm_multi_agents.prompts.loader import load_prompt
from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.utils import merge_state_update_from_tool_messages



def summary_generator_node(
        state: WorkflowState,
) -> Dict[str, Any]:
    if "validate_results" not in state or not state["validate_results"]:
        return {
            "messages": state.get("messages", []),
            "errors": ["summary_generator_node: validate_results is missing"],
        }

    agent_role = "summary_generator"
    default_system = """
    You are a reporting agent specialized in generating summary reports.
    """
    agent = built_graph_agent_by_role(agent_role, default_system=default_system)


    validate_results_str = json.dumps(state["validate_results"], ensure_ascii=False)

    # 获取模板文件路径
    template_path = Path(__file__).resolve().parents[1] / "templates" / "summary_template.md"

    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "summary_generator_prompts.txt"
    user_prompt = load_prompt(prompt_path).format(
        validate_results=validate_results_str,
        # bsm_results=bsm_results_str,
        # greeks_results=greeks_results_str,
        template_path=template_path
    )

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
        tool_names=("generate_summary",),
    )

    return out