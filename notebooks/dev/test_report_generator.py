#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test report_generator_agent to create Word OPA documents
"""
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from bsm_multi_agents.agents.report_generator_agent import report_generator_node

# Example state with markdown and charts
test_state = {
    "messages": [],

    # Path to markdown summary (from summary_generator)
    "report_md": [{
        "file_path": str(Path(__file__).parents[2] / "data" / "output" / "validation_summary_20251117_184530.md")
    }],

    # Chart paths (from chart_generator)
    "chart_results": [
        {
            "status": "success",
            "chart_path": str(Path(__file__).parents[2] / "data" / "output" / "option_prices.png"),
            "description": "Option prices visualization"
        },
        {
            "status": "success",
            "chart_path": str(Path(__file__).parents[2] / "data" / "output" / "greeks_sensitivity.png"),
            "description": "Greeks sensitivity analysis"
        }
    ]
}

print("=" * 80)
print("Testing Report Generator Agent")
print("=" * 80)

# Call the agent
result = report_generator_node(test_state)

print(f"\nStatus: {'✅ Success' if 'report_path' in result else '❌ Failed'}")

if "report_path" in result:
    print(f"Report saved to: {result['report_path']}")

    # Check if file exists
    if Path(result['report_path']).exists():
        print(f"✅ File created successfully!")
        print(f"File size: {Path(result['report_path']).stat().st_size / 1024:.2f} KB")
    else:
        print(f"❌ File not found!")

# Print messages
print("\nMessages:")
for msg in result.get("messages", []):
    print(f"  - {type(msg).__name__}: {msg.content}")

# Print errors if any
if "errors" in result and result["errors"]:
    print("\nErrors:")
    for error in result["errors"]:
        print(f"  - {error}")
