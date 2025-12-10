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
    csv_path = Path(__file__).resolve().parents[2] / "data" / "input" / "dummy_options.csv"

    # Verify CSV file exists
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return

    print(f"\n📂 CSV file: {csv_path}")
    print(f"✅ File exists: {csv_path.exists()}\n")

    # Initialize state
    init_state: WorkflowState = {
        "csv_file_path": str(csv_path),
        "messages": [HumanMessage(content=f"Load CSV from: {csv_path}")],
    }

    # Run the workflow
    print("=" * 80)
    print("Starting Workflow Execution...")
    print("=" * 80)
    print("\nWorkflow stages:")
    print("  1. data_loader → Load CSV data")
    print("  2. calculator → Calculate BSM prices and Greeks")
    print("  3. validator → Validate results")
    print("  4. summary_generator → Generate markdown summary")
    print("  5. chart_generator → Create visualization charts")
    print("  6. report_generator → Generate Word report")
    print("\n" + "=" * 80 + "\n")

    try:
        final_state = app.invoke(
            init_state,
            config={"configurable": {"thread_id": "run-1"}}
        )

        # Print results
        print("\n" + "=" * 80)
        print("Workflow Completed Successfully!")
        print("=" * 80)

        print(f"\n📊 Final State Summary:")
        print(f"  State keys: {list(final_state.keys())}")

        print(f"\n✅ Workflow Progress:")
        stages = [
            ("csv_data", "CSV Data Loaded"),
            ("bsm_results", "BSM Results Calculated"),
            ("greeks_results", "Greeks Calculated"),
            ("validate_results", "Results Validated"),
            ("report_md", "Summary Generated"),
            ("chart_results", "Charts Created"),
            ("report_path", "Word Report Generated"),
        ]

        for key, description in stages:
            status = "✅" if key in final_state else "❌"
            print(f"  {status} {description}")
            if key in final_state and key == "csv_data":
                print(f"      → {len(final_state[key])} rows loaded")
            elif key in final_state and key == "chart_results":
                print(f"      → {len(final_state[key])} charts created")
            elif key in final_state and key == "report_path":
                print(f"      → {final_state[key]}")

        # Print errors if any
        if "errors" in final_state and final_state["errors"]:
            print(f"\n⚠️  Errors encountered:")
            for error in final_state["errors"]:
                print(f"  - {error}")

        # Print final report path
        if "report_path" in final_state:
            print(f"\n" + "=" * 80)
            print(f"📄 Final Report Generated:")
            print(f"   {final_state['report_path']}")
            print("=" * 80)
        else:
            print(f"\n⚠️  Warning: Final report was not generated")

    except Exception as e:
        print(f"\n❌ Error during workflow execution:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()