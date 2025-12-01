# src/bsm_multi_agents/agents/mcp_wrappers.py
import asyncio
import sys
import os
from typing import Any, Dict, Mapping, Annotated
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from langgraph.prebuilt import InjectedState
from bsm_multi_agents.graph.state import WorkflowState


async def call_mcp_tool_async(
    tool_name: str,
    server_script_path: str,
    arguments: Mapping[str, Any],
) -> Any:
    """
    底层 async MCP 调用。
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script_path],
        env=os.environ,
    )
    # server_params = StdioServerParameters(
    #     command="uv",
    #     args=["run", "python", server_script_path],
    # )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments=arguments)


def call_mcp_tool(
    tool_name: str,
    server_script_path: str,
    arguments: Mapping[str, Any],
) -> Any:
    """
    同步包装：方便在 REPL / 单元测试里直接调用 MCP 工具调试。
    """
    return asyncio.run(
        call_mcp_tool_async(tool_name, server_script_path, arguments)
    )


import threading

def run_in_new_loop(coro):
    """
    在单独的线程中运行一个新的事件循环来执行协程。
    这允许在已有的事件循环（如 Jupyter）中同步调用异步代码，而不需要 nest_asyncio。
    """
    result = None
    exception = None

    def runner():
        nonlocal result, exception
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
        except Exception as e:
            exception = e
        finally:
            loop.close()

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if exception:
        raise exception
    return result


def create_mcp_state_tool_wrapper(
    *,
    mcp_tool_name: str,
    server_script_path: str,
    input_arg_map: Dict[str, str],   # state_key -> mcp_arg_name
    output_key: str,                 # MCP 返回值写入 state 的字段名
):
    """
    返回一个 stateful 工具：

      (state) -> state_update

    - 从 state 中按 input_arg_map 取 key，拼成 MCP 调用参数
    - 调 MCP 工具 mcp_tool_name
    - 把 MCP 返回的结果写入 {output_key: result}，作为 state_update 返回
    """

    def state_tool(
        state: Annotated[WorkflowState, InjectedState] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if state is None:
            # Fallback: check if 'state' is in kwargs
            if "state" in kwargs:
                state = kwargs["state"]
            else:
                return {"errors": ["Critical: WorkflowState was not injected into tool."]}

        # 1. 准备 MCP 参数
        tool_args: Dict[str, Any] = {}
        missing_keys = []
        for state_key, mcp_arg_name in input_arg_map.items():
            if state_key not in state or state[state_key] in (None, ""):
                missing_keys.append(state_key)
            else:
                # if isinstance(state[state_key], Path):
                #     state[state_key] = str(state[state_key])
                tool_args[mcp_arg_name] = state[state_key]

        if missing_keys:
            err_msg = f"MCP tool '{mcp_tool_name}' missing state keys: {missing_keys}"
            return {"errors": [err_msg]}

        # 2. 调用 MCP 工具
        try:
            # 使用 run_in_new_loop 在新线程中运行异步 MCP 调用
            tool_res = run_in_new_loop(
                call_mcp_tool_async(
                    tool_name=mcp_tool_name,
                    server_script_path=server_script_path,
                    arguments=tool_args,
                )
            )
        except Exception as e:
            err_msg = f"MCP tool '{mcp_tool_name}' call failed: {e}"
            return {"errors": [err_msg]}

        # 3. 解析 MCP 返回结果
        # fastmcp 的返回通常有 structuredContent / content
        result_value = None

        # 优先 structuredContent["result"]
        sc = getattr(tool_res, "structuredContent", None)
        if isinstance(sc, dict) and "result" in sc:
            result_value = sc["result"]

        # 其次 TextContent.text
        if result_value is None:
            content = getattr(tool_res, "content", None)
            if content:
                first = content[0]
                if hasattr(first, "text"):
                    result_value = first.text

        # 再不行就 str()
        if result_value is None:
            result_value = str(tool_res)

        # 4. 返回 state_update
        return {output_key: result_value}

    return state_tool