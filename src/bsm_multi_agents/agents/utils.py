from typing import Dict, Any, Iterable
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage



# def create_mcp_state_tool_wrapper(
#     *,  # 推荐用 keyword-only，避免顺序错
#     mcp_tool_name: str,
#     server_script_path: str,
#     input_arg_map: Dict[str, str],
#     output_key: str,
# ):
#     """
#     返回一个函数： state -> state_update

#     - 只从 state 读数据（根据 input_arg_map）
#     - 调用 MCP 工具 mcp_tool_name
#     - 把结果写到 {output_key: result} 里返回
#     """

#     def state_tool(
#         state: Annotated[dict, InjectedState]
#     ) -> Dict[str, Any]:
#         # 1. 从 state 中准备 MCP 调用参数
#         tool_args = {}
#         for state_key, mcp_arg_name in input_arg_map.items():
#             if state_key not in state:
#                 # 这里直接返回 errors，方便 Agent 看到并决定怎么做
#                 return {"errors": [f"Missing required state key '{state_key}'"]}
#             tool_args[mcp_arg_name] = state[state_key]

#         # 2. 调用 MCP（第二层 helper）
#         try:
#             result = call_mcp_tool(
#                 tool_name=mcp_tool_name,
#                 server_script_path=server_script_path,
#                 arguments=tool_args,
#             )
#         except Exception as e:
#             return {"errors": [f"MCP call failed: {e}"]}

#         # 3. 处理 MCP 返回结果
#         if isinstance(result, str) and result.startswith("Error"):
#             return {"errors": [result]}

#         # 4. 返回 state_update（第三层）
#         return {output_key: result}

#     return state_tool











def get_tool_result_from_messages(messages, tool_name):
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            try:
                return {'result': msg.content}
            except json.JSONDecodeError:
                return {"Error": "Failed to decode JSON from tool output"}
            break
    return {"Error": "Tool message not found"}

def print_resp(resp):
    step_num = 1
    for message in resp["messages"]:
        if isinstance(message, HumanMessage):
            print(f"Step {step_num} - inputs:")
            print(f"   {message.content[:200]}..." if len(message.content) > 200 else f"   {message.content}")
            print()
            step_num += 1

        elif isinstance(message, AIMessage):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # Agent 决定调用工具
                print(f"Step {step_num} - Agent decide tools used:")
                for tool_call in message.tool_calls:
                    print(f"   Tool name: {tool_call['name']}")
                    print(f"   Tool parameters: {tool_call['args']}")
                print()
                step_num += 1
            elif message.content:
                print(f"Step {step_num} - Agent outputs:")
                print(f"   {message.content}")
                print()
                step_num += 1

        elif isinstance(message, ToolMessage):
            print(f"Step {step_num} - outputs:")
            print(f"   Tool name: {message.name}")
            # result_preview = message.content[:300] + "..." if len(message.content) > 300 else message.content
            result_preview = message.content
            print(f"   Outputs: {result_preview}")
            print()
            step_num += 1

    print(f"\n{'='*80}")
    print(f"Final outputs:")
    print(f"{'='*80}\n")
    print(resp["messages"][-1].content)
