from typing import List
from langchain_core.tools import BaseTool

from core.bsm_calculator import csv_loader

def get_data_loader_tools() -> List[BaseTool]:
    """
    Tools for Agent 1: Data Loader
    - Load CSV data
    """
    return [csv_loader]