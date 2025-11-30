from __future__ import annotations
import operator
from typing import TypedDict, List, Dict, Any, Literal, Annotated, Sequence
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph.message import add_messages

from bsm_multi_agents.tools.utils import JSON_STR


class WorkflowState(TypedDict, total=False):
    # Conversation messages (for agents to read/append)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Business data
    csv_file_path: str
    bsm_results_path: str
    greeks_results_path: str
    validate_results_path: str
    report_md_path: str
    report_charts_path: str
    report_path: str

    # Workflow control
    errors: Annotated[List[str], operator.add]
    remaining_steps: Annotated[int, operator.add]

