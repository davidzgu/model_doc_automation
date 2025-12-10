from pathlib import Path
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.tools.summary_generator_tools import generate_summary



def summary_generator_node(
        state: WorkflowState,
) -> Dict[str, Any]:
    """
    Generate a markdown summary report from validation results.

    Args:
        state: Workflow state containing validate_results

    Returns:
        Updated state with messages and report_md
    """
    if "validate_results" not in state or not state["validate_results"]:
        return {
            "messages": state.get("messages", []),
            "errors": ["summary_generator_node: validate_results is missing"],
        }

    # Get template path
    template_path = str(Path(__file__).resolve().parents[1] / "templates" / "summary_template.md")

    # Call the tool directly
    result_json = generate_summary.invoke({
        "validate_results": json.dumps(state["validate_results"], ensure_ascii=False),
        "template_path": template_path,
        "save_md": True
    })

    # Parse the result (format: {"state_update": {"report_md": "...", "report_path": "..."}})
    result_dict = json.loads(result_json)
    state_update = result_dict.get("state_update", {})

    # Build messages
    merged_messages = list(state.get("messages", []))
    merged_messages.append(HumanMessage(content="Generate markdown summary report from validation results."))

    # Extract report info
    report_md = state_update.get("report_md")
    report_path = state_update.get("report_path")

    if report_md:
        if report_path:
            merged_messages.append(AIMessage(
                content=f"✅ Summary generated successfully!\nSaved to: {report_path}"
            ))
        else:
            merged_messages.append(AIMessage(
                content=f"✅ Summary generated successfully (in-memory)"
            ))
    else:
        error_msg = state_update.get("errors", ["Unknown error"])[0] if "errors" in state_update else "Unknown error"
        merged_messages.append(AIMessage(content=f"❌ Summary generation failed: {error_msg}"))

    # Build output state
    out = {"messages": merged_messages}

    # Add report info to state
    if report_md:
        # Store as list of dicts for consistency with other agents
        out["report_md"] = [{
            "content": report_md,
            "file_path": report_path if report_path else None
        }]

    if "errors" in state_update:
        out["errors"] = state.get("errors", []) + state_update["errors"]

    return out