# src/bsm_multi_agents/agents/mcp_client.py
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
    Low-level async MCP tool call.
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
    # When running inside Jupyter, `sys.stderr`/`sys.stdout` are ipykernel
    # streams that do not implement a usable `fileno()` which breaks
    # subprocess creation on Windows. Provide a real file object (devnull)
    # for the child process stderr to avoid this issue.
    devnull = open(os.devnull, "w", encoding=server_params.encoding)
    try:
        async with stdio_client(server_params, errlog=devnull) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments=arguments)
    finally:
        try:
            devnull.close()
        except Exception:
            pass


def call_mcp_tool(
    tool_name: str,
    server_script_path: str,
    arguments: Mapping[str, Any],
) -> Any:
    """
    Synchronous wrapper: convenient for calling MCP tools directly
    from a REPL or unit tests.
    """
    return asyncio.run(
        call_mcp_tool_async(tool_name, server_script_path, arguments)
    )


import threading

def run_in_new_loop(coro):
    """
    Run a new event loop in a dedicated thread to execute a coroutine.

    This allows synchronously calling async code from an environment
    where an event loop is already running (e.g., Jupyter) without
    using nest_asyncio.
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
    input_arg_map: Dict[str, str],   
    output_key: str,                
):
    """
    Return a stateful tool function:

      (state) -> state_update

    - Read keys from the state according to `input_arg_map` to build MCP arguments.
    - Call the MCP tool `mcp_tool_name`.
    - Put the MCP result into `{output_key: result}` and return it as the state_update.
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

        # 1. Prepare MCP arguments
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

        # 2. Call MCP tool
        try:
            # Use run_in_new_loop to run async MCP call in a new thread
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

        # 3. Parse MCP return result
        result_value = None

        # First try structuredContent["result"]
        sc = getattr(tool_res, "structuredContent", None)
        if isinstance(sc, dict) and "result" in sc:
            result_value = sc["result"]

        # Next try TextContent.text
        if result_value is None:
            content = getattr(tool_res, "content", None)
            if content:
                first = content[0]
                if hasattr(first, "text"):
                    result_value = first.text

        # If all else fails, just str()
        if result_value is None:
            result_value = str(tool_res)

        # 4. Return state_update
        return {output_key: result_value}

    return state_tool