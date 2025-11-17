from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from bsm_multi_agents.graph.agent_graph import build_app, WorkflowState


app = build_app()
csv_path = Path.cwd().parents[1] / "data" / "input" / "dummy_options.csv"
init_state: WorkflowState = {
    "csv_file_path": str(csv_path),
    "messages": [HumanMessage(content=f"Load CSV from: {csv_path}")],
}
final = app.invoke(
    init_state,
    config={"configurable": {"thread_id": "run-1"}}
)