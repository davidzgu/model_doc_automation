# -*- coding: utf-8 -*-
"""
Agent node implementations for the multi-agent workflow.

Each agent is a callable class that processes the state and returns updates.
"""
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Any

from .state import OptionAnalysisState
from .tools import (
    get_agent1_tools,
    get_agent2_tools,
    get_agent3_tools,
    get_agent4_tools,
    get_agent5_tools,
    get_agent6_tools
)


class Agent1_DataLoader:
    """Agent 1: Load CSV data containing option parameters"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_agent1_tools(),
            state_schema=OptionAnalysisState
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Load CSV data"""
        try:
            csv_path = state["csv_file_path"]

            # Create task message
            task_message = HumanMessage(content=f"""
Load the option data from the CSV file at: {csv_path}

Use the csv_loader tool to read the CSV file. Return the data in JSON format.
""")

            # Invoke agent
            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract output from last message
            if result.get("messages"):
                # Look for tool messages with CSV data
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'name') and msg.name == 'csv_loader':
                        import json
                        try:
                            csv_data = json.loads(msg.content)
                            return {
                                "csv_data": csv_data,
                                "agent1_status": "completed",
                                "current_agent": "agent1",
                                "workflow_status": "in_progress"
                            }
                        except:
                            pass

            return {
                "agent1_status": "failed",
                "current_agent": "agent1",
                "errors": ["Failed to load CSV data"]
            }

        except Exception as e:
            return {
                "agent1_status": "error",
                "current_agent": "agent1",
                "errors": [f"Agent 1 error: {str(e)}"]
            }


class Agent2_Calculator:
    """Agent 2: Calculate BSM prices and Greeks"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_agent2_tools(),
            state_schema=OptionAnalysisState
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Calculate option prices and Greeks"""
        try:
            csv_data = state.get("csv_data")
            if not csv_data:
                return {
                    "agent2_status": "failed",
                    "errors": ["No CSV data available from Agent 1"]
                }

            # Create task message
            task_message = HumanMessage(content=f"""
Calculate Black-Scholes-Merton option prices and Greeks for the following data:

{csv_data}

Use batch_bsm_calculator for price calculation and sensitivity_test for one sample option to get Greeks data.

Return:
1. A markdown table with all option prices
2. Greeks sensitivity analysis for the first option
""")

            # Invoke agent
            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract calculation results
            calculation_results = None
            greeks_data = None

            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name'):
                    if msg.name == 'batch_bsm_calculator':
                        calculation_results = msg.content
                    elif msg.name == 'sensitivity_test':
                        import json
                        try:
                            greeks_data = json.loads(msg.content)
                        except:
                            pass

            return {
                "calculation_results": calculation_results or "No calculation results",
                "greeks_data": greeks_data,
                "agent2_status": "completed",
                "current_agent": "agent2",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "agent2_status": "error",
                "current_agent": "agent2",
                "errors": [f"Agent 2 error: {str(e)}"]
            }


class Agent3_Tester:
    """Agent 3: Run validation tests"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_agent3_tools(),
            state_schema=OptionAnalysisState
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Run tests on calculation results"""
        try:
            csv_data = state.get("csv_data")
            if not csv_data:
                return {
                    "agent3_status": "skipped",
                    "test_results": {"status": "skipped", "message": "No data to test"}
                }

            task_message = HumanMessage(content=f"""
Run validation tests on the option calculations.

Data: {csv_data}

Use the following tools:
1. run_greeks_validation_test - to validate Greeks calculations
2. run_sensitivity_analysis_test - to test sensitivity analysis

Report the test results.
""")

            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract test results
            test_results = {"tests_run": [], "overall_status": "unknown"}

            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name') and 'test' in msg.name:
                    import json
                    try:
                        test_data = json.loads(msg.content)
                        test_results["tests_run"].append({
                            "test_name": msg.name,
                            "result": test_data
                        })
                    except:
                        pass

            if test_results["tests_run"]:
                test_results["overall_status"] = "completed"

            return {
                "test_results": test_results,
                "agent3_status": "completed",
                "current_agent": "agent3",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "agent3_status": "error",
                "current_agent": "agent3",
                "errors": [f"Agent 3 error: {str(e)}"]
            }


class Agent4_SummaryWriter:
    """Agent 4: Write textual summary"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_agent4_tools(),
            state_schema=OptionAnalysisState
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Write summary based on all previous results"""
        try:
            calculation_results = state.get("calculation_results", "No results")
            test_results = state.get("test_results", {})

            task_message = HumanMessage(content=f"""
Write a comprehensive summary of the option analysis.

Calculation Results:
{calculation_results}

Test Results:
{test_results}

Create a professional markdown summary with the following sections:
1. Overview
2. Key Findings
3. Calculation Summary
4. Test Results Summary
5. Conclusions

Keep it concise and clear.
""")

            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Get summary from last AI message
            summary_text = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'content') and len(msg.content) > 100:
                    summary_text = msg.content
                    break

            return {
                "summary_text": summary_text or "Summary generation failed",
                "agent4_status": "completed",
                "current_agent": "agent4",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "agent4_status": "error",
                "current_agent": "agent4",
                "errors": [f"Agent 4 error: {str(e)}"]
            }


class Agent5_ChartGenerator:
    """Agent 5: Generate charts and visualizations"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_agent5_tools(),
            state_schema=OptionAnalysisState
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Generate charts"""
        try:
            csv_data = state.get("csv_data")
            greeks_data = state.get("greeks_data")

            task_message = HumanMessage(content=f"""
Generate visualization charts for the option analysis.

CSV Data: {csv_data}
Greeks Data: {greeks_data}

Use create_summary_charts tool to generate both price and Greeks charts.
Save them to outputs/charts/ directory.
""")

            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract chart paths
            charts = []
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name') and 'chart' in msg.name:
                    import json
                    try:
                        chart_info = json.loads(msg.content)
                        if chart_info.get("status") == "success":
                            if "charts" in chart_info:
                                charts.extend(chart_info["charts"])
                            elif "chart_path" in chart_info:
                                charts.append(chart_info["chart_path"])
                    except:
                        pass

            return {
                "charts": charts,
                "agent5_status": "completed",
                "current_agent": "agent5",
                "workflow_status": "in_progress"
            }

        except Exception as e:
            return {
                "agent5_status": "error",
                "current_agent": "agent5",
                "errors": [f"Agent 5 error: {str(e)}"]
            }


class Agent6_ReportAssembler:
    """Agent 6: Assemble final report"""

    def __init__(self, llm):
        self.agent = create_react_agent(
            model=llm,
            tools=get_agent6_tools(),
            state_schema=OptionAnalysisState
        )

    def __call__(self, state: OptionAnalysisState) -> Dict[str, Any]:
        """Assemble final HTML report"""
        try:
            summary_text = state.get("summary_text", "No summary")
            charts = state.get("charts", [])

            task_message = HumanMessage(content=f"""
Assemble the final HTML report combining the summary and charts.

Summary:
{summary_text}

Charts: {charts}

Use assemble_html_report tool to create a complete HTML report.
Save it to outputs/report.html
""")

            result = self.agent.invoke({
                **state,
                "messages": [task_message]
            })

            # Extract report path
            report_path = None
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, 'name') and msg.name == 'assemble_html_report':
                    import json
                    try:
                        report_info = json.loads(msg.content)
                        if report_info.get("status") == "success":
                            report_path = report_info.get("report_path")
                    except:
                        pass

            return {
                "final_report_path": report_path or "Report generation failed",
                "agent6_status": "completed",
                "current_agent": "agent6",
                "workflow_status": "completed"
            }

        except Exception as e:
            return {
                "agent6_status": "error",
                "current_agent": "agent6",
                "workflow_status": "failed",
                "errors": [f"Agent 6 error: {str(e)}"]
            }
