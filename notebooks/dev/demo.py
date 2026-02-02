import importlib
from pathlib import Path
from pprint import pprint
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import time

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from bsm_multi_agents.graph import agent_graph
importlib.reload(agent_graph)
from bsm_multi_agents.graph.agent_graph import build_app, WorkflowState

"""Run the complete workflow."""

def generate_report(path):
    # Build the application graph
    #print("=" * 80)
    yield("Building Agent Graph...")
    time.sleep(0.1)
    #print("=" * 80)
    app = build_app()
    #yield from build_app()
    # path = "dummy_options.csv"
    csv_path = Path.cwd().parents[1] / "data" / "input" / path
    init_state: WorkflowState = {
        "csv_file_path": str(csv_path),
        "messages": [HumanMessage(content=f"Load CSV from: {csv_path}")],
    }
    # Run the workflow
    #print("=" * 80)
    yield("Starting Workflow Execution...")
    time.sleep(2)
    #print("=" * 80)
    yield("\nWorkflow stages:")
    yield("  1. data_loader → Load CSV data")
    time.sleep(5)
    yield("  2. calculator → Calculate BSM prices and Greeks")
    time.sleep(5)
    yield("  3. validator → Validate results")
    time.sleep(5)
    yield("  4. summary_generator → Generate markdown summary")
    time.sleep(5)
    yield("  5. chart_generator → Create visualization charts")
    time.sleep(5)
    yield("  6. report_generator → Generate Word report")
    #print("\n" + "=" * 80 + "\n")

    final_state = app.invoke(
        init_state,
        config={"configurable": {"thread_id": "run-1"}}
    )

    # Print results
    #print("\n" + "=" * 80)
    yield("Workflow Completed Successfully!")
    time.sleep(0.1)
    #print("=" * 80)

    yield(f"\n📊 Final State Summary:")
    yield(f"  State keys: {list(final_state.keys())}")
    time.sleep(0.1)
    st.write(f"\n✅ Workflow Progress:")
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
        yield(f"  {status} {description}")
        if key in final_state and key == "csv_data":
            yield(f"      → {len(final_state[key])} rows loaded")
        elif key in final_state and key == "chart_results":
            yield(f"      → {len(final_state[key])} charts created")
        elif key in final_state and key == "report_path":
            yield(f"REPORT_PATH:{final_state[key]}")

    # Print errors if any
    if "errors" in final_state and final_state["errors"]:
        st.write(f"\n⚠️  Errors encountered:")
        for error in final_state["errors"]:
            st.write(f"  - {error}")

    # Print final report path
    if "report_path" in final_state:
        print(f"\n" + "=" * 80)
        st.write(f"📄 Final Report Generated:")
        st.write(f"   {final_state['report_path']}")
        #print("=" * 80)
        print(final_state['report_path'])
    else:
        st.write(f"\n⚠️  Warning: Final report was not generated")

# def your_function_stream():
#     for i in range(10):
#         print(f"Step {i}")
#         yield f"Step {i}"
#         time.sleep(1)

if __name__ == "__main__":
    generate_report("dummy_options.csv")