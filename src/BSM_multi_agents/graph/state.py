from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Literal
from langchain_core.messages import AnyMessage


class WorkflowState(TypedDict, total=False):
    # 对话消息（给各 agent 读取/追加）
    messages: List[AnyMessage]

    # 业务数据
    csv_file_path: str
    csv_data: List[Dict[str, Any]]
    bsm_results: List[Dict[str, Any]]
    greeks_results: List[Dict[str, Any]]
    validate_results: List[Dict[str, Any]]
    report_md: List[Dict[str, Any]]
    

    # 流程控制
    next_agent: Literal["data_loader", "calculator", "end"]
    errors: List[str]

