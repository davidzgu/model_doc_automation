import asyncio
import os
import importlib.util
import inspect
from typing import Dict, Any, Iterable, List

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from bsm_multi_agents.agents.mcp_adapter import (
    list_mcp_tools_sync, 
    mcp_tool_to_langchain_tool
)

def extract_mcp_content(tool_res) -> str:
    """Helper to extract text content from MCP CallToolResult"""
    result_value = None
    
    # Try structuredContent["result"]
    sc = getattr(tool_res, "structuredContent", None)
    if isinstance(sc, dict) and "result" in sc:
        result_value = sc["result"]

    # Try TextContent.text
    if result_value is None:
        content = getattr(tool_res, "content", None)
        if content:
            # content is a list of TextContent or ImageContent
            texts = []
            for item in content:
                if hasattr(item, "text"):
                    texts.append(item.text)
            if texts:
                result_value = "\n".join(texts)

    # Fallback
    if result_value is None:
        result_value = str(tool_res)
        
    return str(result_value)




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


def load_local_tools_from_file(file_path: str) -> List[StructuredTool]:
    """
    Load LangChain StructuredTools from a local Python file.
    Each function in the file is converted into a tool.
    """
    if not os.path.exists(file_path):
        return []

    module_name = os.path.basename(file_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tools = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # Exclude private functions or imports
        if not name.startswith("_") and func.__module__ == module_name:
            tools.append(StructuredTool.from_function(func))
    
    return tools


def load_tools_from_mcp_and_local(server_path:str, local_tool_paths:List[str]):
    mcp_tools = list_mcp_tools_sync(server_path)

    langchain_tools = [mcp_tool_to_langchain_tool(t, server_path) for t in mcp_tools]
    
    for path in local_tool_paths:
        local_tools = load_local_tools_from_file(path)
        langchain_tools.extend(local_tools)

    return langchain_tools