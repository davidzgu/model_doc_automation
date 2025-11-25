from bsm_multi_agents.agents.calculator_agent import calculator_node
from bsm_multi_agents.graph.state import WorkflowState
from langchain_core.messages import HumanMessage
import json

# Mock state with csv_data
mock_csv_data = [
    {"option_type": "call", "S": 100, "K": 105, "T": 1, "r": 0.05, "sigma": 0.2},
    {"option_type": "put", "S": 100, "K": 95, "T": 0.5, "r": 0.05, "sigma": 0.2}
]
mock_state: WorkflowState = {
    "csv_data": json.dumps(mock_csv_data), # Tool expects JSON string or list/dict, let's pass list directly as it handles it
    "messages": [],
    "thread_id": "test-run",
    "remaining_steps": 10
}
# Actually the tool load_json_as_df handles list of dicts too.
mock_state["csv_data"] = mock_csv_data

print("Running calculator_node with mock state...")
try:
    result = calculator_node(mock_state)
    print("\nResult messages:")
    for msg in result.get("messages", []):
        print(f"Type: {type(msg)}, Content: {msg.content}")
        if hasattr(msg, "tool_calls"):
             print(f"Tool Calls: {msg.tool_calls}")

    if "errors" in result:
        print(f"\nErrors: {result['errors']}")
    else:
        print("\nNo errors found.")

except Exception as e:
    print(f"\nException occurred: {e}")
