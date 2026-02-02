from pathlib import Path
import json
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.tools.report_generator_tools import create_word_report


def report_generator_node(
        state: WorkflowState,
) -> Dict[str, Any]:
    """
    Generate a Word OPA report from markdown summary and charts.

    Args:
        state: Workflow state containing:
            - report_md: List of markdown file info or path string
            - chart_results: List of chart info dicts with 'chart_path'

    Returns:
        Updated state with messages and report_path
    """

    # Extract markdown path
    markdown_path = None
    if "report_md" in state and state["report_md"]:
        report_md = state["report_md"]
        if isinstance(report_md, list) and len(report_md) > 0:
            # Could be list of dicts or list of strings
            if isinstance(report_md[0], dict):
                markdown_path = report_md[0].get("file_path") or report_md[0].get("path")
            else:
                markdown_path = report_md[0]
        elif isinstance(report_md, str):
            markdown_path = report_md

    if not markdown_path:
        return {
            "messages": state.get("messages", []),
            "errors": ["report_generator_node: report_md is missing or invalid"],
        }

    # Extract chart paths
    chart_paths = []
    if "chart_results" in state and state["chart_results"]:
        chart_results = state["chart_results"]
        if isinstance(chart_results, list):
            for chart in chart_results:
                if isinstance(chart, dict):
                    path = chart.get("chart_path")
                    if path:
                        chart_paths.append(path)
                elif isinstance(chart, str):
                    chart_paths.append(chart)

    # Set output path
    output_dir = Path(__file__).resolve().parents[3] / "data" / "output"
    timestamp = Path(markdown_path).stem.split('_')[-1] if '_' in Path(markdown_path).stem else "latest"
    output_path = str(output_dir / f"OPA_Report_{timestamp}.docx")

    # Call the tool directly
    result_json = create_word_report.invoke({
        "markdown_path": markdown_path,
        "chart_paths": json.dumps(chart_paths) if chart_paths else "[]",
        "output_path": output_path,
        "title": "OPA Report"
    })

    # Parse result
    result_dict = json.loads(result_json)

    # Build messages
    merged_messages = list(state.get("messages", []))
    merged_messages.append(HumanMessage(content="Generate Word OPA report from markdown and charts."))

    if result_dict.get("status") == "success":
        doc_path = result_dict.get("document_path")
        merged_messages.append(AIMessage(
            content=f"✅ Word report generated successfully!\nDocument saved to: {doc_path}"
        ))
    else:
        error_msg = result_dict.get("message", "Unknown error")
        merged_messages.append(AIMessage(content=f"❌ Report generation failed: {error_msg}"))

    # Build output state
    out = {"messages": merged_messages}

    # Add report path to state if successful
    if result_dict.get("status") == "success":
        out["report_path"] = result_dict.get("document_path")
    else:
        out["errors"] = state.get("errors", []) + [result_dict.get("message", "Report generation failed")]

    return out
