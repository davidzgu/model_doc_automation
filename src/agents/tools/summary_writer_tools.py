from typing import List
from langchain_core.tools import BaseTool


from src.agents.tools.report_tools import load_template

def get_summary_writer_tools() -> List[BaseTool]:
    """
    Tools for Summary Writer
    - Load templates
    """
    return [load_template]