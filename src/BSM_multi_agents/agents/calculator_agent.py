from pathlib import Path
from typing import Dict, Any, List

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool, StructuredTool

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.mcp_client import create_mcp_state_tool_wrapper
from bsm_multi_agents.config.llm_config import get_llm
from bsm_multi_agents.prompts.loader import load_prompt


# def build_mcp_calculator_tools(server_path: str) -> List[StructuredTool]:
#     """
#     把 MCP 的 calculate_bsm_to_file / calculate_greeks_to_file
#     封装成带 InjectedState 的 LangChain Tool，给 ReAct agent 用。
#     """
#     # 1) BSM 工具
#     bsm_state_fn = create_mcp_state_tool_wrapper(
#         mcp_tool_name="calculate_bsm_to_file",
#         server_script_path=server_path,
#         input_arg_map={
#             # state["csv_file_path"] -> MCP 参数 input_path
#             "csv_file_path": "input_path",
#             "output_dir": "output_dir",
#         },
#         output_key="bsm_results_path",
#     )
#     bsm_tool = StructuredTool.from_function(
#         name="mcp_bsm_to_file",
#         func=bsm_state_fn,
#         description=(
#             "Call MCP tool 'calculate_bsm_to_file'. "
#             "Read 'csv_file_path' and 'output_dir' from workflow state, "
#             "write 'bsm_results_path' (result file path) back to state."
#         ),
#     )

#     # 2) Greeks 工具
#     greeks_state_fn = create_mcp_state_tool_wrapper(
#         mcp_tool_name="calculate_greeks_to_file",
#         server_script_path=server_path,
#         input_arg_map={
#             "csv_file_path": "input_path",
#             "output_dir": "output_dir",
#         },
#         output_key="greeks_results_path",
#     )
#     greeks_tool = StructuredTool.from_function(
#         name="mcp_greeks_to_file",
#         func=greeks_state_fn,
#         description=(
#             "Call MCP tool 'calculate_greeks_to_file'. "
#             "Read 'csv_file_path' and 'output_dir' from workflow state, "
#             "write 'greeks_results_path' (result file path) back to state."
#         ),
#     )

#     return [bsm_tool, greeks_tool]

def calculator_agent_node(current_state: WorkflowState) -> WorkflowState:
    """
    LangGraph node：内部是一个 ReAct agent。

    Agent 目标：
      - 在给定 csv_file_path 和 output_dir 的前提下，
        根据 prompt 自动决定调用 BSM / Greeks 工具，并把结果路径写回 state。
    """
    errors = current_state.get("errors", [])

    if "csv_file_path" not in current_state or not current_state["csv_file_path"]:
        errors.append("calculator_node: csv_file_path is missing")
        current_state["errors"] = errors
        return current_state

    # if "output_dir" not in current_state or not current_state["output_dir"]:
    #     current_state["output_dir"] = str(
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
        user_prompt_template.format(csv_file_path=current_state["csv_file_path"])
        if "{csv_file_path}" in user_prompt_template
        else user_prompt_template
    )

    # agent
    llm = get_llm()
    system_msg = SystemMessage(content=system_prompt)
    human_msg = HumanMessage(content=formatted_user_prompt)
    ai_msg = llm.invoke([system_msg, human_msg])

    messages = list(current_state.get("messages", []))
    messages.extend([system_msg, human_msg, ai_msg])
    current_state["messages"] = messages

    current_state["errors"] = errors
    return current_state

    # tools = build_mcp_calculator_tools(server_path)

    
def calculator_tool_node(current_state: WorkflowState) -> WorkflowState:
    """
    Tool node：纯工具逻辑，不用 LLM。

    - 从 state 中读取 server_path / csv_file_path / output_dir
    - 通过 MCP 调用 calculate_bsm_to_file / calculate_greeks_to_file
    - 把生成的结果路径写入：
        state["bsm_results_path"]
        state["greeks_results_path"]
    - 同时收集任何 errors 写回 state["errors"]
    """
    errors = current_state.get("errors", [])

    if "csv_file_path" not in current_state or not current_state["csv_file_path"]:
        errors.append("calculator_node: csv_file_path is missing")
        current_state["errors"] = errors
        return current_state

    if "server_path" not in current_state or not current_state["server_path"]:
        errors.append("calculator_tool_node: server_path is missing")
        current_state["errors"] = errors
        return current_state
    
    server_path = current_state["server_path"]

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

    bsm_update = bsm_state_fn(state=current_state)
    greeks_update = greeks_state_fn(state=current_state)

    # bsm_update / greeks_update 形如：
    #   {"bsm_results_path": "..."} 或 {"errors": [...]}
    if "errors" in bsm_update:
        errors.extend(bsm_update["errors"])
    if "errors" in greeks_update:
        errors.extend(greeks_update["errors"])

    if "bsm_results_path" in bsm_update:
        current_state["bsm_results_path"] = bsm_update["bsm_results_path"]
    if "greeks_results_path" in greeks_update:
        current_state["greeks_results_path"] = greeks_update["greeks_results_path"]

    current_state["errors"] = errors
    return current_state


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    csv_file_path = str(project_root / "data/input/dummy_options.csv")
    output_dir = str(project_root / "data/cache")
    server_path = str(project_root / "src" / "bsm_multi_agents" / "mcp" / "server.py")
    current_state = WorkflowState(
        csv_file_path=csv_file_path, 
        output_dir=output_dir, 
        server_path=server_path
    )
    result_state = calculator_node(current_state)
    print(result_state)