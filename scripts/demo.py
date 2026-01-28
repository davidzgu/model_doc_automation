import os
from pathlib import Path
from bsm_multi_agents.graph.agent_graph import build_app
from bsm_multi_agents.graph.state import WorkflowState
import importlib

def main():
    project_path = Path.cwd()
    csv_file_path = os.path.join(project_path, "data/input/simulated_equity_options.csv")
    output_dir = str(project_path / "data/cache")
    server_path = str(project_path / "src/bsm_multi_agents/mcp/server.py")
    local_tool_folder_path = os.path.join(project_path, "src/bsm_multi_agents/tools")
    final_report_path = str(project_path / "data/output/final_report.docx")

    app = build_app()

    initial_state = WorkflowState(
        csv_file_path=csv_file_path,
        output_dir=output_dir,
        server_path=server_path,
        local_tool_folder_path=local_tool_folder_path,
        final_report_path=final_report_path,
        errors=[],
        messages=[],
        # "remaining_steps": 10,
    )

    config = {"configurable": {"thread_id": "demo_thread_1"}}

    print("Starting Multi-Agent Workflow...\n")

    final_state = app.invoke(
        initial_state,
        config=config,
        recursion_limit=20
    )

    print("\nWorkflow Complete.")

if __name__ == "__main__":
    main()
    