import json
import re
from typing import Dict, Any, Iterable
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from mcp.client import ClientSession, StdioServerParameters

def create_mcp_tool_wrapper(tool_name: str, server_script_path: str, input_arg_map: Dict[str, str], output_key: str):
    """
    创建一个同步的 LangChain Tool，内部调用异步的 MCP Client。
    
    Args:
        tool_name: MCP Server 上的工具名称 (e.g., "calculate_bsm_to_file")
        server_script_path: MCP Server 脚本的绝对路径
        input_arg_map: 映射 State 中的 key 到 MCP 工具的参数名 (e.g., {"csv_file_path": "input_path"})
        output_key: 将 MCP 工具返回的结果保存到 State 的哪个 key (e.g., "bsm_results_path")
    """
    async def _run_mcp_tool(kwargs):
        # 启动 MCP Server 进程
        server_params = StdioServerParameters(
            command="python", 
            args=[server_script_path]
        )
        async with ClientSession(server_params) as session:
            await session.initialize()
            # 调用工具
            return await session.call_tool(tool_name, arguments=kwargs)

    def wrapper(state: dict):
        # 1. 准备参数：从 State 中提取数据
        tool_args = {}
        for state_key, tool_arg_name in input_arg_map.items():
            if state_key not in state:
                return f"Error: Missing required state key '{state_key}'"
            tool_args[tool_arg_name] = state.get(state_key)
            
        # 2. 调用 MCP 工具 (同步运行异步代码)
        try:
            result_path = asyncio.run(_run_mcp_tool(tool_args))
            
            # 3. 处理结果：检查是否出错
            if isinstance(result_path, str) and result_path.startswith("Error"):
                return {"errors": [result_path]}
            
            # 4. 更新 State：保存返回的路径
            return {output_key: result_path}
            
        except Exception as e:
            return {"errors": [f"MCP Tool execution failed: {str(e)}"]}

    return wrapper

# def get_tool_result_from_messages(messages, tool_name):
#     for msg in reversed(messages):
#         if isinstance(msg, ToolMessage) and msg.name == tool_name:
#             try:
#                 return {'result': msg.content}
#             except json.JSONDecodeError:
#                 return {"Error": "Failed to decode JSON from tool output"}
#             break
#     return {"Error": "Tool message not found"}

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
