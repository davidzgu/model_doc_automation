from typing import List
from langchain_core.tools import BaseTool

from core.bsm_calculator import (
    greeks_calculator,
    sensitivity_test,
    batch_bsm_calculator,
)


def get_calculation_tools() -> List[BaseTool]:
    """
    Tools for Calculation Agent
    - Calculate BSM prices
    - Calculate Greeks
    - Run sensitivity analysis
    """
    return [
        batch_bsm_calculator,
        greeks_calculator,
        sensitivity_test
    ]