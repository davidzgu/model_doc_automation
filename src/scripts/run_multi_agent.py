# -*- coding: utf-8 -*-
"""
Main entry point for the 6-agent option analysis workflow.

Usage:
    python -m src.run_multi_agent [CSV_FILE_PATH]

Example:
    python -m src.run_multi_agent ../inputs/dummy_options.csv
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from multi_agent.workflow import create_multi_agent_workflow, print_workflow_status
from multi_agent.state import OptionAnalysisState
from llm import get_llm


def main(csv_path: str = None):
    """
    Run the complete 6-agent workflow for option analysis.

    Args:
        csv_path: Path to the CSV file containing option data.
                 If None, uses default path from inputs/dummy_options.csv
    """
    # Determine CSV path
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[2] / "inputs" / "dummy_options.csv"
    else:
        csv_path = Path(csv_path).resolve()

    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return

    print("\n" + "=" * 80)
    print("🚀 MULTI-AGENT OPTION ANALYSIS WORKFLOW")
    print("=" * 80)
    print(f"\n📁 Input CSV: {csv_path}")
    print(f"📊 Workflow: 6 agents in sequence")
    print("\nAgent Pipeline:")
    print("  1️⃣  Data Loader      → Load CSV data")
    print("  2️⃣  Calculator       → Calculate BSM prices & Greeks")
    print("  3️⃣  Tester          → Run validation tests")
    print("  4️⃣  Summary Writer   → Generate text summary")
    print("  5️⃣  Chart Generator  → Create visualizations")
    print("  6️⃣  Report Assembler → Create final HTML report")
    print("\n" + "=" * 80 + "\n")

    # Initialize LLM
    print("🔧 Initializing language model...")
    llm = get_llm()

    # Create workflow
    print("🔧 Building multi-agent workflow...")
    workflow = create_multi_agent_workflow(llm)

    # Prepare initial state
    initial_state: OptionAnalysisState = {
        "csv_file_path": str(csv_path),
        "messages": [],
        "csv_data": None,
        "agent1_status": None,
        "calculation_results": None,
        "greeks_data": None,
        "agent2_status": None,
        "test_results": None,
        "agent3_status": None,
        "summary_text": None,
        "agent4_status": None,
        "charts": None,
        "chart_descriptions": None,
        "agent5_status": None,
        "final_report_path": None,
        "final_report_html": None,
        "agent6_status": None,
        "current_agent": None,
        "workflow_status": "started",
        "errors": []
    }

    # Run workflow
    print("▶️  Starting workflow execution...\n")
    print("-" * 80)

    try:
        config = {"configurable": {"thread_id": "option-analysis-1"}}
        result = workflow.invoke(initial_state, config)

        print("-" * 80)
        print("\n✅ Workflow execution completed!\n")

        # Print final status
        print_workflow_status(result)

        # Print key results
        print("\n" + "=" * 80)
        print("📋 RESULTS SUMMARY")
        print("=" * 80 + "\n")

        if result.get("calculation_results"):
            print("📊 Calculation Results:")
            print(result["calculation_results"][:500] + "..." if len(result["calculation_results"]) > 500 else result["calculation_results"])
            print()

        if result.get("test_results"):
            print("🧪 Test Results:")
            print(f"   Status: {result['test_results'].get('overall_status', 'Unknown')}")
            print(f"   Tests Run: {len(result['test_results'].get('tests_run', []))}")
            print()

        if result.get("charts"):
            print("📈 Charts Generated:")
            for chart in result["charts"]:
                if isinstance(chart, dict):
                    print(f"   - {chart.get('chart_path', chart)}")
                else:
                    print(f"   - {chart}")
            print()

        if result.get("final_report_path"):
            print("📄 Final Report:")
            print(f"   {result['final_report_path']}")
            print()
            print(f"   Open in browser: file://{result['final_report_path']}")

        print("\n" + "=" * 80)
        print("✨ Analysis complete!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Workflow execution failed!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Get CSV path from command line argument or use default
    csv_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv_path_arg)
