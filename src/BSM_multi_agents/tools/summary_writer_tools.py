from typing import List
from langchain_core.tools import BaseTool


from src.core.summary_generator import generate_summary

def get_summary_writer_tools() -> List[BaseTool]:
    """
    Tools for Summary Writer
    - Load templates
    """
    return [generate_summary]