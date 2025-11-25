from __future__ import annotations
import operator
from typing import TypedDict, List, Dict, Any, Literal, Annotated, Sequence
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph.message import add_messages

from bsm_multi_agents.tools.utils import JSON_STR


class WorkflowState(TypedDict, total=False):
    # Conversation messages (for agents to read/append)
    # messages: List[AnyMessage]
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Business data
    csv_file_path: str
    csv_data: JSON_STR
    bsm_results: JSON_STR
    greeks_results: JSON_STR
    validate_results: JSON_STR
    # report_md: List[Dict[str, Any]]
    # chart_results: List[Dict[str, Any]]
    # report_path: str

    # # Workflow control
    # next_agent: Literal["data_loader", "calculator", "end"]
    errors: Annotated[List[str], operator.add]
    remaining_steps: Annotated[int, operator.add]

