from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.pricing_calculator_node import (
    pricing_calculator_agent_node,
    pricing_calculator_tool_node,
)

from bsm_multi_agents.agents.pricing_validator_node import (
    pricing_validator_agent_node,
    pricing_validator_tool_node,
)

from bsm_multi_agents.agents.report_generator_node import (
    report_generator_agent_node,
)


def should_continue_for_pricing_calculator(state):
    messages = state["messages"]
    if not messages:
        return END
        
    last_msg = messages[-1]
    
    # 1. If it's an Agent message (AIMessage) with tool_calls -> Go to Tool
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "pricing_calculator_tool"
    
    # 2. If it's a Tool Execution message (ToolMessage)
    #    Check for errors to decide if we should retry
    if isinstance(last_msg, ToolMessage):
         if last_msg.content.startswith("Error"):
             # OPTIONAL: Check iteration count to prevent infinite loop
             return "pricing_calculator_agent"
         
         # Success -> Finish
         return "pricing_validator_agent"
    
    # Default
    return "pricing_validator_agent"

def should_continue_for_pricing_validator(state):
    messages = state["messages"]
    if not messages:
        return END
        
    last_msg = messages[-1]
    
    # 1. If it's an Agent message (AIMessage) with tool_calls -> Go to Tool
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "pricing_validator_tool"
    
    # 2. If it's a Tool Execution message (ToolMessage)
    #    Check for errors to decide if we should retry
    if isinstance(last_msg, ToolMessage):
         if last_msg.content.startswith("Error"):
             # OPTIONAL: Check iteration count to prevent infinite loop
             return "pricing_validator_agent"
         
         # Success -> Finish
         return "report_generator_agent"
    
    # Default
    return "report_generator_agent"


graph = StateGraph(WorkflowState)
graph.add_node("pricing_calculator_agent", pricing_calculator_agent_node)
graph.add_node("pricing_calculator_tool", pricing_calculator_tool_node)
graph.add_node("pricing_validator_agent", pricing_validator_agent_node)
graph.add_node("pricing_validator_tool", pricing_validator_tool_node)
graph.add_node("report_generator_agent", report_generator_agent_node)

graph.add_edge(START, "pricing_calculator_agent")
graph.add_edge("pricing_calculator_agent", "pricing_calculator_tool")
graph.add_conditional_edges(
    "pricing_calculator_tool",
    should_continue_for_pricing_calculator,
    {
        "pricing_calculator_agent": "pricing_calculator_agent", # Retry
        "pricing_validator_agent": "pricing_validator_agent" # Success
    }
)
graph.add_edge("pricing_validator_agent", "pricing_validator_tool")
graph.add_conditional_edges(
    "pricing_validator_tool",
    should_continue_for_pricing_validator,
    {
        "pricing_validator_agent": "pricing_validator_agent", # Retry
        "report_generator_agent": "report_generator_agent" # Success
    }
)
graph.add_edge("report_generator_agent", END)

# Persistence for notebooks
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

def get_graph_image():
    """Returns the graph visualization as a PNG image for display in notebooks."""
    return app.get_graph().draw_mermaid_png()