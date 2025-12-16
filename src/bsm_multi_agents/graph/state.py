from __future__ import annotations
import operator
from typing import TypedDict, List, Dict, Any, Literal, Annotated, Sequence
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph.message import add_messages


class WorkflowState(TypedDict, total=False):
    # Conversation messages (for agents to read/append)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Generic tool outputs
    tool_outputs: Dict[str, Any]

    # core paths
    csv_file_path: str
    output_dir: str
    server_path: str 

    # pricing artifacts
    bsm_results_path: str
    greeks_results_path: str

    # validation artifacts
    validate_results_path: str

    # text artifacts
    md_summary_text: str        
    md_summary_path: str       
    report_plan_text: str       
    report_path: str      

    # charts
    chart_paths: List[str]      # 多张图片路径或目录 

    # control flags (由 OrchestratorAgent 决定，可选）
    run_pricing: bool
    run_validation: bool
    run_md_summary: bool
    run_charts: bool
    run_report: bool

    # Workflow control
    errors: Annotated[List[str], operator.add]
    remaining_steps: Annotated[int, operator.add]

