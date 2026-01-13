# src/bsm_multi_agents/agents/mcp_client.py
import asyncio
import json
import sys
import os
from typing import Any, Dict, List, Type, Mapping, Annotated
from pathlib import Path
import threading

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from langgraph.prebuilt import InjectedState
from langchain_core.tools import StructuredTool
from bsm_multi_agents.graph.state import WorkflowState


from pydantic import create_model, BaseModel, Field


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
    Synchronous wrapper: convenient for calling MCP tools directly
    from a REPL or unit tests.
    """
    return run_in_new_loop(
        call_mcp_tool_async(tool_name, server_script_path, arguments)
    )
    # return asyncio.run(
    #     call_mcp_tool_async(tool_name, server_script_path, arguments)
    # )




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


async def list_mcp_tools_async(server_script_path: str):
    """
    Connect to the MCP server and list available tools.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script_path],
        env=os.environ,
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools

def list_mcp_tools_sync(server_script_path: str):
    """
    Synchronous wrapper to list MCP tools.
    """
    return run_in_new_loop(list_mcp_tools_async(server_script_path))

def _json_schema_to_pydantic(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Convert a JSON schema dict to a Pydantic model.
    This is a simplified converter.
    """
    fields = {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for field_name, field_info in properties.items():
        field_type = str
        # Simple type mapping
        msg_type = field_info.get("type", "string")
        if msg_type == "integer":
            field_type = int
        elif msg_type == "number":
            field_type = float
        elif msg_type == "boolean":
            field_type = bool
        elif msg_type == "object":
            field_type = dict
        elif msg_type == "array":
            field_type = list
        
        description = field_info.get("description", "")
        
        if field_name in required:
            fields[field_name] = (field_type, Field(..., description=description))
        else:
            fields[field_name] = (field_type, Field(default=None, description=description))

    return create_model(name + "Input", **fields)

def mcp_tool_to_langchain_tool(mcp_tool: Any, server_script_path: str) -> StructuredTool:
    """
    Convert an MCP tool definition to a LangChain StructuredTool.
    The tool will execute using call_mcp_tool.
    """
    tool_name = mcp_tool.name
    tool_description = mcp_tool.description or ""
    input_schema = mcp_tool.inputSchema
    
    # Create Pydantic model from schema
    args_schema = _json_schema_to_pydantic(tool_name, input_schema)

    def tool_func(**kwargs):
        # We don't use 'state' here because LangGraph's tool node (prebuilt or custom)
        # usually passes args directly.
        # But our custom pricing_calculator_tool_node logic might need adjustment.
        # Actually, for standard tool calling, we just receive args.
        return call_mcp_tool(
            tool_name=tool_name,
            server_script_path=server_script_path,
            arguments=kwargs
        )

    return StructuredTool.from_function(
        func=tool_func,
        name=tool_name,
        description=tool_description,
        args_schema=args_schema,
    )