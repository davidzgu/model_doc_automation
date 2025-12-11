from pathlib import Path
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from bsm_multi_agents.graph.state import WorkflowState
from bsm_multi_agents.agents.mcp_client import create_mcp_state_tool_wrapper


def pricing_validator_tool_node(
    state: WorkflowState,
) -> WorkflowState:
    """
    Tool node: pure tool logic, no LLM involved.

    - Read `server_path`, `greeks_results_path`, and `output_dir` from the state.
    - Call `validate_greeks_to_file` via the MCP server.
    - Write the generated result file paths into:
        state["validate_results_path"]
    - Collect any errors and write them back to state["errors"].
    """

    errors = state.get("errors", [])
    if "greeks_results_path" not in state or not state["greeks_results_path"]:
        errors.append("pricing_validator_agent_node: greeks_results_path is missing")
        state["errors"] = errors
        return state
    
    server_path = state["server_path"]


    validate_greeks_state_fn = create_mcp_state_tool_wrapper(
        mcp_tool_name="validate_greeks_to_file",
        server_script_path=server_path,
        input_arg_map={
            "greeks_results_path": "input_path",
            "output_dir": "output_dir",
        },
        output_key="validate_results_path",
    )

    validate_update = validate_greeks_state_fn(state=state)

    if "errors" in validate_update:
        errors.extend(validate_update["errors"])

    if "validate_results_path" in validate_update:
        state["validate_results_path"] = validate_update["validate_results_path"]
    
    state["errors"] = errors
    return state

