from __future__ import annotations

from typing import TypedDict, Literal, Any, Dict, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.data_loader_agent import data_loader_node
from bsm_multi_agents.agents.calculator_agent import calculator_node
from bsm_multi_agents.agents.validator_agent import validator_node
from bsm_multi_agents.agents.summary_generator_agent import summary_generator_node
from bsm_multi_agents.agents.chart_generator_agent import chart_generator_node
from bsm_multi_agents.agents.report_generator_agent import report_generator_node





def router(state: WorkflowState) -> Literal[
    "data_loader", "calculator", "validator",
    "summary_generator","chart_generator","report_generator",
    "end"
]:
    # If a node has explicitly set next_agent, prioritize that
    if state.get("next_agent"):
        next_node = state["next_agent"]
        print(f"[Router] next_agent explicitly set to: {next_node}")
        return next_node  # type: ignore[return-value]

    # Sequential workflow logic based on state completeness
    if "csv_data" not in state:
        print("[Router] csv_data missing → routing to data_loader")
        return "data_loader"
    if "bsm_results" not in state or "greeks_results" not in state:
        print(f"[Router] bsm_results or greeks_results missing → routing to calculator")
        return "calculator"
    if "validate_results" not in state:
        print("[Router] validate_results missing → routing to validator")
        return "validator"

    # After validator, we need both summary_generator and chart_generator
    # They can run in parallel, so we check which ones are missing
    has_summary = "report_md" in state and state.get("report_md")
    has_charts = "chart_results" in state and state.get("chart_results")

    # If neither summary nor charts exist, prioritize summary_generator
    if not has_summary:
        print("[Router] report_md missing → routing to summary_generator")
        return "summary_generator"
    if not has_charts:
        print("[Router] chart_results missing → routing to chart_generator")
        return "chart_generator"

    # If both summary and charts exist, check if we need report_generator
    if "report_path" not in state:
        print("[Router] report_path missing → routing to report_generator")
        return "report_generator"

    # Everything is complete
    print("[Router] All stages complete → routing to END")
    return "end"


# ---------- 组装 Graph ----------
def build_app():
    graph = StateGraph(WorkflowState)

    # 注册节点
    graph.add_node("data_loader", data_loader_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("validator", validator_node)
    graph.add_node("summary_generator", summary_generator_node)
    graph.add_node("chart_generator", chart_generator_node)
    graph.add_node("report_generator", report_generator_node)

    # Set entry point to data_loader
    graph.set_entry_point("data_loader")

    # All nodes use router to determine next step
    for node in ["data_loader", "calculator", "validator", "summary_generator", "chart_generator", "report_generator"]:
        graph.add_conditional_edges(
            node,
            router,
            {
                "data_loader": "data_loader",
                "calculator": "calculator",
                "validator": "validator",
                "summary_generator": "summary_generator",
                "chart_generator": "chart_generator",
                "report_generator": "report_generator",
                "end": END,
            },
        )

    # 可选：内存型检查点，便于多轮/断点调试
    return graph.compile(checkpointer=MemorySaver())

