from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.mcp_client import create_mcp_state_tool_wrapper
# from bsm_multi_agents.config.llm_config import get_llm
from bsm_multi_agents.config.llm_config import get_llm
from bsm_multi_agents.prompts.loader import load_prompt



def pricing_calculator_agent_node(state: WorkflowState) -> WorkflowState:
    """
    LangGraph node: LLM planning / explanation step.

    Objectives:
    - Given `csv_file_path` and `output_dir`,
        generate an explanation and plan for running BSM and Greeks calculations.
    - This node does not call tools directly; it only appends messages
        to the shared workflow state.
    """
    errors = state.get("errors", [])

    if "csv_file_path" not in state or not state["csv_file_path"]:
        errors.append("pricing_calculator_agent_node: csv_file_path is missing")
        state["errors"] = errors
        return state

    # if "output_dir" not in state or not state["output_dir"]:
    #     state["output_dir"] = str(
    #         Path(__file__).resolve().parents[1] / "data" / "cache"
    #     )


    # prompts
    system_prompt = """
    You are a quantitative calculator agent.
    You have access to tools that call an external MCP server.
    Each tool reads input file paths from the shared workflow state
    and writes result file paths back into the state.
    Do NOT ask for raw CSV content. Always work with paths only.
    You do not need to provide any arguments to the tools. Just call them with {}.
    """

    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "calculator_prompts.txt"
    )
    user_prompt_template = load_prompt(prompt_path)
    formatted_user_prompt = (
        user_prompt_template.format(csv_file_path=state["csv_file_path"])
        if "{csv_file_path}" in user_prompt_template
        else user_prompt_template
    )

    # agent
    llm = get_llm()
    system_msg = SystemMessage(content=system_prompt)
    human_msg = HumanMessage(content=formatted_user_prompt)
    ai_msg = llm.invoke([system_msg, human_msg])

    messages = list(state.get("messages", []))
    messages.extend([system_msg, human_msg, ai_msg])
    state["messages"] = messages

    state["errors"] = errors
    return state

    # tools = build_mcp_calculator_tools(server_path)

    
def pricing_calculator_tool_node(state: WorkflowState) -> WorkflowState:
    """
    Tool node: pure tool logic, no LLM involved.

    - Read `server_path`, `csv_file_path`, and `output_dir` from the state.
    - Call `calculate_bsm_to_file` and `calculate_greeks_to_file` via the MCP server.
    - Write the generated result file paths into:
        state["bsm_results_path"]
        state["greeks_results_path"]
    - Collect any errors and write them back to state["errors"].
    """
    errors = state.get("errors", [])

    if "csv_file_path" not in state or not state["csv_file_path"]:
        errors.append("pricing_calculator_tool_node: csv_file_path is missing")
        state["errors"] = errors
        return state

    if "server_path" not in state or not state["server_path"]:
        errors.append("pricing_calculator_tool_node: server_path is missing")
        state["errors"] = errors
        return state
    
    server_path = state["server_path"]

    bsm_state_fn = create_mcp_state_tool_wrapper(
        mcp_tool_name="calculate_bsm_to_file",
        server_script_path=server_path,
        input_arg_map={
            "csv_file_path": "input_path",
            "output_dir": "output_dir",
        },
        output_key="bsm_results_path",
    )

    greeks_state_fn = create_mcp_state_tool_wrapper(
        mcp_tool_name="calculate_greeks_to_file",
        server_script_path=server_path,
        input_arg_map={
            "csv_file_path": "input_path",
            "output_dir": "output_dir",
        },
        output_key="greeks_results_path",
    )

    bsm_update = bsm_state_fn(state=state)
    greeks_update = greeks_state_fn(state=state)

    # bsm_update / greeks_update 形如：
    #   {"bsm_results_path": "..."} 或 {"errors": [...]}
    if "errors" in bsm_update:
        errors.extend(bsm_update["errors"])
    if "errors" in greeks_update:
        errors.extend(greeks_update["errors"])

    if "bsm_results_path" in bsm_update:
        state["bsm_results_path"] = bsm_update["bsm_results_path"]
    if "greeks_results_path" in greeks_update:
        state["greeks_results_path"] = greeks_update["greeks_results_path"]

    state["errors"] = errors
    return state


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    csv_file_path = str(project_root / "data/input/dummy_options.csv")
    output_dir = str(project_root / "data/cache")
    server_path = str(project_root / "src" / "bsm_multi_agents" / "mcp" / "server.py")
    state = WorkflowState(
        csv_file_path=csv_file_path, 
        output_dir=output_dir, 
        server_path=server_path
    )
    result_state = pricing_calculator_agent_node(state)
    print(result_state)