from typing import List
from langchain_core.tools import BaseTool

from core.chart_tools import create_option_price_chart, create_greeks_chart, create_summary_charts


def get_chart_generator_tools() -> List[BaseTool]:
    """
    Tools for Chart Generator
    - Create price charts
    - Create Greeks charts
    - Create summary charts
    """
    return [
        create_option_price_chart,
        create_greeks_chart,
        create_summary_charts
    ]