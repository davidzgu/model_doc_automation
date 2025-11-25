from .data_loader_tools import csv_loader
from .bsm_calculator_tools import (
    # bsm_calculator,
    batch_bsm_calculator,
    # greeks_calculator,
    batch_greeks_calculator,
)
from .validator_tools import (
    validate_greeks_rules,
    batch_greeks_validator
)
from .summary_generator_tools import generate_summary
from .chart_generator_tools import (
    create_option_price_chart,
    create_greeks_chart,
    create_summary_charts
)

from .tool_registry import REGISTRY

def get_tools_for_role(role: str):
    return REGISTRY.by_role(role)

def get_tools_by_tags(*tags: str):
    return REGISTRY.by_tag(*tags)

def get_tools_by_name(name: str):
    return REGISTRY.get_by_name(name)