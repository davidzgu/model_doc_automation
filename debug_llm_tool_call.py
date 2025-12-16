
import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Define a simple tool
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

def debug_llm():
    model_name = "qwen3:32b" # The model user is using
    print(f"Testing model: {model_name}")
    
    try:
        llm = ChatOllama(model=model_name, temperature=0)
        
        # Bind tool
        llm_with_tools = llm.bind_tools([add])
        
        # Invoke
        messages = [HumanMessage(content="What is 11 + 22? Use the add tool.")]
        print("Invoking LLM...")
        ai_msg = llm_with_tools.invoke(messages)
        
        print(f"\nModel Response Type: {type(ai_msg)}")
        print(f"Content: '{ai_msg.content}'")
        print(f"Tool Calls: {ai_msg.tool_calls}")
        print(f"Response Metadata: {ai_msg.response_metadata}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_llm()
