from typing import List
from langchain_core.tools import BaseTool

from src.agents.tools.report_tools import assemble_html_report


def get_report_assembler_tools() -> List[BaseTool]:
    """
    Tools for Agent 6: Report Assembler
    - Assemble final HTML report
    """
    return [assemble_html_report]