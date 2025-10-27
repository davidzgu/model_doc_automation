# -*- coding: utf-8 -*-
"""
Tool allocation for each agent in the multi-agent workflow.

Each agent has access to a specific set of tools relevant to its task.
"""
from typing import List
from langchain_core.tools import BaseTool

# Import all available tools
from src.bsm_utils import csv_loader, batch_bsm_calculator, greeks_calculator, sensitivity_test
from src.test_tools import run_greeks_validation_test, run_sensitivity_analysis_test
from src.chart_tools import create_option_price_chart, create_greeks_chart, create_summary_charts
from src.report_tools import load_template, assemble_html_report


def get_agent1_tools() -> List[BaseTool]:
    """
    Tools for Agent 1: Data Loader
    - Load CSV data
    """
    return [csv_loader]


def get_agent2_tools() -> List[BaseTool]:
    """
    Tools for Agent 2: Calculation Agent
    - Calculate BSM prices
    - Calculate Greeks
    - Run sensitivity analysis
    """
    return [
        batch_bsm_calculator,
        greeks_calculator,
        sensitivity_test
    ]


def get_agent3_tools() -> List[BaseTool]:
    """
    Tools for Agent 3: Testing Agent
    - Run Greeks validation tests
    - Run sensitivity analysis tests
    """
    return [
        run_greeks_validation_test,
        run_sensitivity_analysis_test
    ]


def get_agent4_tools() -> List[BaseTool]:
    """
    Tools for Agent 4: Summary Writer
    - Load templates
    """
    return [load_template]


def get_agent5_tools() -> List[BaseTool]:
    """
    Tools for Agent 5: Chart Generator
    - Create price charts
    - Create Greeks charts
    - Create summary charts
    """
    return [
        create_option_price_chart,
        create_greeks_chart,
        create_summary_charts
    ]


def get_agent6_tools() -> List[BaseTool]:
    """
    Tools for Agent 6: Report Assembler
    - Assemble final HTML report
    """
    return [assemble_html_report]
