from pathlib import Path
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.tools.chart_generator_tools import create_summary_charts


def chart_generator_node(
        state: WorkflowState,
) -> Dict[str, Any]:

    if "bsm_results" not in state or not state["bsm_results"]:
        return {
            "messages": state.get("messages", []),
            "errors": ["chart_generator_node: bsm_results is missing"],
        }

    if "greeks_results" not in state or not state["greeks_results"]:
        return {
            "messages": state.get("messages", []),
            "errors": ["chart_generator_node: greeks_results is missing"],
        }

    output_dir = str(Path(__file__).resolve().parents[3] / "data" / "output")

    # Call the tool directly
    result_json = create_summary_charts.invoke({
        "bsm_results": json.dumps(state["bsm_results"], ensure_ascii=False),
        "greeks_results": json.dumps(state["greeks_results"], ensure_ascii=False),
        "output_dir": output_dir,
    })

    # Parse the JSON string returned by the tool
    result_dict = json.loads(result_json)

    # Build messages
    merged_messages = list(state.get("messages", []))
    merged_messages.append(HumanMessage(content="Generate visualization charts for the analysis results."))

    # Add AI response message based on result status
    if result_dict.get("status") == "success":
        charts = result_dict.get("charts", [])
        charts_info = "\n".join([
            f"  - {chart['description']}: {chart['chart_path']}"
            for chart in charts
        ])
        merged_messages.append(AIMessage(
            content=f"✅ Chart generation successful! Generated {result_dict.get('total_charts', 0)} charts:\n{charts_info}"
        ))
    else:
        error_msg = result_dict.get("message", "Unknown error")
        merged_messages.append(AIMessage(content=f"❌ Chart generation failed: {error_msg}"))

    # Build output state
    out = {"messages": merged_messages}

    # Add chart info to state if successful
    if result_dict.get("status") == "success":
        out["chart_results"] = result_dict.get("charts", [])
    else:
        out["errors"] = state.get("errors", []) + [result_dict.get("message", "Chart generation failed")]

    return out
