# -*- coding: utf-8 -*-
"""
Test validation tools for the Tester Agent.

Imports validation tools from the consolidated bsm_validator module.
"""
from typing import List
from langchain_core.tools import BaseTool

# Import from the simplified validator module
from src.core.bsm_validator import (
    batch_greeks_validator,
    validate_put_call_parity,
    validate_sensitivity
)


def get_tester_tools() -> List[BaseTool]:
    """
    Return validation tools for the Tester Agent.

    Tools:
    1. batch_greeks_validator - Validates Greeks for all options
    2. validate_put_call_parity - Tests put-call parity for pairs
    3. validate_sensitivity - Validates sensitivity analysis
    """
    return [
        batch_greeks_validator,
        validate_put_call_parity,
        validate_sensitivity
    ]