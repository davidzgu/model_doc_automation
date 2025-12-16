from typing import Dict, Any, Iterable
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


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
