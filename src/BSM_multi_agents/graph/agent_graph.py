from __future__ import annotations

from typing import TypedDict, Literal, Any, Dict, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.data_loader_agent import data_loader_node
from bsm_multi_agents.agents.calculator_agent import calculator_node
from bsm_multi_agents.agents.validator_agent import validator_node





def router(state: WorkflowState) -> Literal["data_loader", "calculator", "validator", "end"]:
    # 如果上游/某节点已经明确设定了 next_agent，则优先遵循
    if state.get("next_agent"):
        return state["next_agent"]  # type: ignore[return-value]

    # 否则按“完成度”推进
    if "csv_data" not in state:
        return "data_loader"
    if "bsm_results" not in state:
        return "calculator"
    if "validate_results" not in state:
        return "validator"
    return "end"


# ---------- 组装 Graph ----------
def build_app():
    graph = StateGraph(WorkflowState)

    # 注册节点
    graph.add_node("data_loader", data_loader_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("validator", validator_node)

    # 入口设为 data_loader（如果你想根据状态动态进入，也可以先设一个固定入口再条件跳转）
    graph.set_entry_point("data_loader")

    # 各节点跑完后都交给 router 决定下一跳
    for node in ["data_loader", "calculator", "validator"]:
        graph.add_conditional_edges(
            node,
            router,
            {
                "data_loader": "data_loader",
                "calculator": "calculator",
                "validator": "validator",
                "end": END,
            },
        )

    # 可选：内存型检查点，便于多轮/断点调试
    return graph.compile(checkpointer=MemorySaver())

