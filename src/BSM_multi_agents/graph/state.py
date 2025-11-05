# -*- coding: utf-8 -*-
"""
State definition for the multi-agent option analysis workflow.

This module defines the shared state schema that flows through all 6 agents.
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
from langchain_core.messages import BaseMessage


class OptionAnalysisState(TypedDict):
    """
    Shared state for the 6-agent workflow.

    Data flows sequentially:
    Input → Agent1 → Agent2 → Agent3 → Agent4 → Agent5 → Agent6 → Output
    """

    # Messages list - accumulates with add operator
    messages: Annotated[List[BaseMessage], add]

    # ===== Input Fields =====
    csv_file_path: str
    """Path to the input CSV file containing option data"""

    # ===== Agent 1 Output: Data Loader =====
    csv_data: Optional[Dict[str, Any]]
    """Loaded CSV data in dict format from Agent 1"""

    data_loader_agent_status: Optional[str]
    """Status of Agent 1 execution"""

    # ===== Agent 2 Output: Calculation =====
    calculation_results: Optional[str]
    """BSM prices and Greeks calculation results from Agent 2 (markdown table)"""

    greeks_data: Optional[Dict[str, Any]]
    """Detailed Greeks data in dict format for charting"""

    sensitivity_data: Optional[Dict[str, Any]]
    """Detailed Greeks data in dict format for charting"""

    calculator_agent_status: Optional[str]
    """Status of Agent 2 execution"""

    # ===== Agent 3 Output: Testing =====
    test_results: Optional[Dict[str, Any]]
    """Test results from Agent 3 (pass/fail status, details)"""

    tester_agent_status: Optional[str]
    """Status of Agent 3 execution"""

    # ===== Agent 4 Output: Summary Writer =====
    summary_text: Optional[str]
    """Written summary in markdown format from Agent 4"""

    summarty_writer_agnet_status: Optional[str]
    """Status of Agent 4 execution"""

    # ===== Agent 5 Output: Chart Generator =====
    charts: Optional[List[str]]
    """List of generated chart file paths from Agent 5"""

    chart_descriptions: Optional[Dict[str, str]]
    """Descriptions of each chart"""

    chart_generator_agent_status: Optional[str]
    """Status of Agent 5 execution"""

    # ===== Agent 6 Output: Report Assembler =====
    final_report_path: Optional[str]
    """Path to the final assembled report from Agent 6"""

    final_report_html: Optional[str]
    """HTML content of the final report"""

    report_assenbler_agent_status: Optional[str]
    """Status of Agent 6 execution"""

    # ===== Workflow Metadata =====
    current_agent: Optional[str]
    """Name of the currently executing agent"""

    workflow_status: str
    """Overall workflow status: started, in_progress, completed, failed"""

    errors: Annotated[List[str], add]
    """List of errors encountered during execution"""
