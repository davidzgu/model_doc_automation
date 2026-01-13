#!/usr/bin/env python
"""
Run the complete BSM multi-agent workflow.

This script demonstrates how to use the agent graph to:
1. Load CSV data
2. Calculate BSM and Greeks
3. Validate results
4. Generate summary and charts
5. Create final Word report
"""

from pathlib import Path
from langchain_core.messages import HumanMessage
from bsm_multi_agents.graph.agent_graph import build_app, WorkflowState


def main():
    """Run the complete workflow."""

    # Build the application graph
    print("=" * 80)
    print("Building Agent Graph...")
    print("=" * 80)
    app = build_app()

    
    # Set up CSV path (same as in Jupyter notebook)
    project_path = Path(__file__).resolve().parents[2]
    csv_file_path = str(project_path / "data/input/dummy_options.csv")
    output_dir = str(project_path / "data/cache")
    server_path = str(project_path / "src/bsm_multi_agents/mcp/server.py")
    local_tool_paths = [os.path.join(
        project_path, "src/bsm_multi_agents/tools/my_add.py"
    )]
    final_report_path = str(project_path / "data/cache/final_report.docx")

    # Verify CSV file exists
    if not csv_file_path.exists():
        print(f"❌ Error: CSV file not found at {csv_file_path}")
        return

    print(f"\n📂 CSV file: {csv_file_path}")
    print(f"✅ File exists: {csv_path.exists()}\n")

    # Initialize state
    initial_state = WorkflowState(
        csv_file_path=csv_file_path,
        output_dir=output_dir,
        server_path=server_path,
        local_tool_paths=local_tool_paths,
        final_report_path=final_report_path,
        errors=[],
        messages=[],
        # "remaining_steps": 10,
    )
    config = {"configurable": {"thread_id": "demo_thread_1"}}

    # Run the workflow
    
    try:
        final_state = app.invoke(
            initial_state,
            config=config
        )
    
    except Exception as e:
        print(f"\n❌ Error during workflow execution:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return None

    return final_state

if __name__ == "__main__":
    main()