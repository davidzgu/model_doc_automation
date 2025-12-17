
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root / "src"))

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.pricing_calculator_node import pricing_calculator_agent_node
from langchain_core.messages import AIMessage

def test_dynamic_tool_loading():
    print("Testing dynamic tool loading...")
    
    # Paths
    tool_path = str(project_root / "src/bsm_multi_agents/tools/add.py")
    csv_file_path = str(project_root / "data/input/dummy_options.csv")
    output_dir = str(project_root / "data/cache")
    server_path = str(project_root / "src/bsm_multi_agents/mcp/server.py")
    
    # Ensure dirs exist
    os.makedirs(output_dir, exist_ok=True)
    
    # State
    state = WorkflowState(
        csv_file_path=csv_file_path,
        output_dir=output_dir,
        server_path=server_path,
        tool_path=[tool_path],
        errors=[],
        messages=[]
    )
    
    # Run Agent Node
    print(f"Loading tools from: {tool_path}")
    final_state = pricing_calculator_agent_node(state)
    
    if final_state["errors"]:
        print("Errors encountered:", final_state["errors"])
        return
    
    # Check if tools are bound and visible to LLM
    # In pricing_calculator_agent_node, the last message in state is the AI Response
    ai_msg = final_state["messages"][-1]
    
    print("\nAgent Response:")
    print(ai_msg.content)
    
    # We can't easily verify tool *use* without actually letting it run, 
    # but we can check if it at least chose to call 'my_add' if prompted.
    # Let's try prompting specifically for the 'my_add' tool.
    
    state["messages"].append(HumanMessage(content="Please use the 'my_add' tool to add 5 and 10."))
    final_state_2 = pricing_calculator_agent_node(state)
    ai_msg_2 = final_state_2["messages"][-1]
    
    print("\nSecond Agent Response (Prompted for my_add):")
    if hasattr(ai_msg_2, "tool_calls"):
        for tc in ai_msg_2.tool_calls:
            print(f"Tool Call: {tc['name']} with args {tc['args']}")
            if tc['name'] == 'my_add':
                print("SUCCESS: 'my_add' tool was correctly identified and called!")
    else:
        print("No tool calls found in response.")

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    test_dynamic_tool_loading()
