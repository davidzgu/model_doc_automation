from pathlib import Path
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from bsm_multi_agents.config.llm_config import get_llm
from bsm_multi_agents.graph.state import WorkflowState

def orchestrator_agent_node(state: WorkflowState) -> WorkflowState:
    """
    Orchestrator agent:
    - Read input paths and existing artifact paths.
    - Decide which steps to run in this execution.
    - Produce a natural-language plan.
    - Optionally set control flags in the state.
    """
    errors = state.get("errors", [])

    if "csv_file_path" not in state or not state["csv_file_path"]:
        errors.append("orchestrator_agent_node: csv_file_path is missing")
        state["errors"] = errors
        return state

    llm = get_llm()
    system_prompt = """
    You are the orchestration agent for an option pricing validation pipeline.

    Steps:
    - pricing: compute BSM prices and greeks.
    - validation: run validation tests on prices/greeks.
    - md_summary: summarize validation results in Markdown.
    - charts: generate diagnostic charts.
    - report: assemble a final Word report.

    Look at what already exists in the state (non-empty paths) and decide:
    - which steps must run now;
    - which can be skipped (e.g., pricing already done).

    Return a JSON object in your final answer with the fields:
    {
      "run_pricing": true/false,
      "run_validation": true/false,
      "run_md_summary": true/false,
      "run_charts": true/false,
      "run_report": true/false
    }
    """

    snapshot = {
        "csv_file_path": state.get("csv_file_path"),
        "pricing_results_path": state.get("pricing_results_path"),
        "greeks_results_path": state.get("greeks_results_path"),
        "validation_results_path": state.get("validation_results_path"),
        "md_summary_path": state.get("md_summary_path"),
        "chart_paths": state.get("chart_paths"),
        "report_path": state.get("report_path"),
    }

    human_msg = HumanMessage(
        content=f"Current state snapshot:\n{snapshot}\n\nDecide which steps to run."
    )
    system_msg = SystemMessage(content=system_prompt)
    ai_msg = llm.invoke([system_msg, human_msg])

    messages = list(state.get("messages", []))
    messages.extend([system_msg, human_msg, ai_msg])
    state["messages"] = messages

    # parse JSON plan
    try:
        import json
        text = ai_msg.content
        start = text.rfind("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            plan = json.loads(text[start : end + 1])
            for key in [
                "run_pricing",
                "run_validation",
                "run_md_summary",
                "run_charts",
                "run_report",
            ]:
                if key in plan:
                    state[key] = bool(plan[key])
    except Exception as e:
        errors.append(f"orchestrator_agent_node: failed to parse plan: {e}")

    state["errors"] = errors
    return state