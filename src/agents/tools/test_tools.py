from typing import List
from langchain_core.tools import BaseTool

from core.test_tools import run_greeks_validation_test, run_sensitivity_analysis_test

def get_test_tools() -> List[BaseTool]:
    """
    Tools for Testing Agent
    - Run Greeks validation tests
    - Run sensitivity analysis tests
    """
    return [
        run_greeks_validation_test,
        run_sensitivity_analysis_test
    ]