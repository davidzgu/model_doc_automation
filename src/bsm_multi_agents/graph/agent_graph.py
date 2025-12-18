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
    """
    Router for Pricing Calculator.
    - If LLM wants to call tools -> Go to Tool Node.
    - If LLM is done (no tool calls) -> Go to Validator (Next Stage).
    """
    messages = state["messages"]
    last_msg = messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "pricing_calculator_tool"
        
    # No tool calls -> Finished this stage
    return "pricing_validator_agent"


def should_continue_for_pricing_validator(state):
    """
    Router for Pricing Validator.
    - If LLM wants to call tools -> Go to Tool Node.
    - If LLM is done (no tool calls) -> Go to Report Generator (Next Stage).
    """
    messages = state["messages"]
    last_msg = messages[-1]
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "pricing_validator_tool"
        
    return "report_generator_agent"

def build_app():
    

    graph = StateGraph(WorkflowState)
    graph.add_node("pricing_calculator_agent", pricing_calculator_agent_node)
    graph.add_node("pricing_calculator_tool", pricing_calculator_tool_node)
    graph.add_node("pricing_validator_agent", pricing_validator_agent_node)
    graph.add_node("pricing_validator_tool", pricing_validator_tool_node)
    graph.add_node("report_generator_agent", report_generator_agent_node)

    graph.add_edge(START, "pricing_calculator_agent")
    graph.add_conditional_edges(
        "pricing_calculator_agent",
        should_continue_for_pricing_calculator,
        {
            "pricing_calculator_tool": "pricing_calculator_tool", 
            "pricing_validator_agent": "pricing_validator_agent"
        }
    )
    graph.add_edge("pricing_calculator_tool", "pricing_calculator_agent")
    graph.add_conditional_edges(
        "pricing_validator_agent",
        should_continue_for_pricing_validator,
        {
            "pricing_validator_tool": "pricing_validator_tool", 
            "report_generator_agent": "report_generator_agent"
        }
    )
    graph.add_edge("pricing_validator_tool", "pricing_validator_agent")
    graph.add_edge("report_generator_agent", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    return app

def get_graph_image(app):
    """Returns the graph visualization as a PNG image for display in notebooks."""
    return app.get_graph().draw_mermaid_png()