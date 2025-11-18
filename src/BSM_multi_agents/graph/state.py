from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Literal
from langchain_core.messages import AnyMessage


class WorkflowState(TypedDict, total=False):
    # Conversation messages (for agents to read/append)
    messages: List[AnyMessage]

    # Business data
    csv_file_path: str
    csv_data: List[Dict[str, Any]]
    bsm_results: List[Dict[str, Any]]
    greeks_results: List[Dict[str, Any]]
    validate_results: List[Dict[str, Any]]
    report_md: List[Dict[str, Any]]
    chart_results: List[Dict[str, Any]]
    report_path: str

    # Workflow control
    next_agent: Literal["data_loader", "calculator", "end"]
    errors: List[str]

