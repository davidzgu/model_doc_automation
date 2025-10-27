# -*- coding: utf-8 -*-
"""
LangGraph workflow for the 6-agent option analysis system.

Builds a StateGraph with sequential agent execution:
START → Agent1 → Agent2 → Agent3 → Agent4 → Agent5 → Agent6 → END
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import OptionAnalysisState
from .agents import (
    Agent1_DataLoader,
    Agent2_Calculator,
    Agent3_Tester,
    Agent4_SummaryWriter,
    Agent5_ChartGenerator,
    Agent6_ReportAssembler
)


def create_multi_agent_workflow(llm):
    """
    Create the complete 6-agent workflow using LangGraph StateGraph.

    Args:
        llm: Language model instance (e.g., ChatOllama)

    Returns:
        Compiled StateGraph ready for execution
    """

    # Initialize all agents
    agent1 = Agent1_DataLoader(llm)
    agent2 = Agent2_Calculator(llm)
    agent3 = Agent3_Tester(llm)
    agent4 = Agent4_SummaryWriter(llm)
    agent5 = Agent5_ChartGenerator(llm)
    agent6 = Agent6_ReportAssembler(llm)

    # Build the graph
    builder = StateGraph(OptionAnalysisState)

    # Add nodes
    builder.add_node("agent1_data_loader", agent1)
    builder.add_node("agent2_calculator", agent2)
    builder.add_node("agent3_tester", agent3)
    builder.add_node("agent4_summary_writer", agent4)
    builder.add_node("agent5_chart_generator", agent5)
    builder.add_node("agent6_report_assembler", agent6)

    # Add sequential edges (linear workflow)
    builder.add_edge(START, "agent1_data_loader")
    builder.add_edge("agent1_data_loader", "agent2_calculator")
    builder.add_edge("agent2_calculator", "agent3_tester")
    builder.add_edge("agent3_tester", "agent4_summary_writer")
    builder.add_edge("agent4_summary_writer", "agent5_chart_generator")
    builder.add_edge("agent5_chart_generator", "agent6_report_assembler")
    builder.add_edge("agent6_report_assembler", END)

    # Compile with memory for state persistence
    workflow = builder.compile(checkpointer=MemorySaver())

    return workflow


def print_workflow_status(state: OptionAnalysisState):
    """
    Print a summary of the workflow execution status.

    Args:
        state: Current workflow state
    """
    print("\n" + "=" * 80)
    print("WORKFLOW STATUS SUMMARY")
    print("=" * 80)

    agents_status = [
        ("Agent 1 - Data Loader", state.get("agent1_status")),
        ("Agent 2 - Calculator", state.get("agent2_status")),
        ("Agent 3 - Tester", state.get("agent3_status")),
        ("Agent 4 - Summary Writer", state.get("agent4_status")),
        ("Agent 5 - Chart Generator", state.get("agent5_status")),
        ("Agent 6 - Report Assembler", state.get("agent6_status")),
    ]

    for agent_name, status in agents_status:
        status_emoji = {
            "completed": "✅",
            "in_progress": "🔄",
            "pending": "⏳",
            "failed": "❌",
            "error": "⚠️",
            None: "➖"
        }.get(status, "❓")

        print(f"{status_emoji} {agent_name:30} {status or 'Not started'}")

    print("\nOverall Status:", state.get("workflow_status", "unknown"))

    if state.get("errors"):
        print("\nErrors:")
        for error in state["errors"]:
            print(f"  - {error}")

    print("=" * 80 + "\n")
