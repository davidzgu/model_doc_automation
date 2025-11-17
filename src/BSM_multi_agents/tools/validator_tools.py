from typing import Union, Dict, Any, List
import json
import pandas as pd

from langchain.tools import tool

from .tool_registry import register_tool
from .utils import load_json_as_df


@register_tool(tags=["greeks","validate"], roles=["validator"])
@tool("validate_greeks_rules")
def validate_greeks_rules(
        option_type: str,
        price: float, 
        delta: float, 
        gamma: float, 
        vega: float
    ) -> str:
    """
    Validate Greeks against business rules.
    Returns list of validation results.
    """
    validations_result = "passed"
    validations_details = []

    # Rule 1: price > 0
    try:
        price_valid = price > 0
        if not price_valid:
            validations_result = "failed"
            validations_details.append(f"Price {price} is not positive")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Price Error: {str(e)}")

    # Rule 2: delta range
    try:
        if option_type.lower() == 'call':
            delta_valid = 0 <= delta <= 1
            expected_range = "[0, 1]"
        else:
            delta_valid = -1 <= delta <= 0
            expected_range = "[-1, 0]"
        if not delta_valid:
            validations_result = "failed"
            validations_details.append(f"Delta {delta:.4f} outside {expected_range}")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Delta Error: {str(e)}")

    # Rule 3: gamma >= 0
    try:
        gamma_valid = gamma >= 0
        if not gamma_valid:
            validations_result = "failed"
            validations_details.append(f"Gamma {gamma:.4f} is negative")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Gamma Error: {str(e)}")

    # Rule 4: vega >= 0
    try:
        vega_valid = vega >= 0
        if not gamma_valid:
            validations_result = "failed"
            validations_details.append(f"Vega {vega:.4f} is negative")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Vega Error: {str(e)}")

    results = {
        "validations_result": validations_result,
        "validations_details": validations_details
    }
    return json.dumps(results)

@register_tool(tags=["greeks","validate","batch"], roles=["validator"])
@tool("batch_greeks_validator")
def batch_greeks_validator(csv_data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> str:
    """
    Validate Greeks for ALL options from CSV data.

    For each option:
    - Validates: price > 0
    - Validates: delta in [0,1] for calls, [-1,0] for puts
    - Validates: gamma >= 0, vega >= 0

    Args:
        csv_data: JSON string, dict, or list of dicts with columns: option_type, price, delta, gamma, vega

    Returns:
        JSON string with state_update containing validate_results
    """
    try:
        df = load_json_as_df(csv_data)
        if df is False:
            return json.dumps({"state_update": {"errors": [f"csv_data must be a string, dict, or list, got {type(csv_data)}"]}})
        

        required_cols = ['option_type', 'price', 'delta', 'gamma', 'vega']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps({"status": "error", "message": f"Missing columns: {missing}"})

        def calc_row(row):
            res = validate_greeks_rules.invoke(
                {
                    "option_type": row['option_type'], 
                    "price": row['price'], 
                    "delta": row['delta'], 
                    "gamma": row['gamma'], 
                    "vega": row['vega'], 
                }
            )
            try:
                d = json.loads(res)             # res 是 JSON 字符串 → 解析成 dict
            except Exception as e:
                d = {"error": f"invalid JSON from greeks_calculator: {e}"}
            return d
        
        expanded = df.apply(calc_row, axis=1).apply(pd.Series)
        result_cols = ['validations_result','validations_details']
        for col in result_cols:
            if col not in expanded:
                expanded[col] = pd.NA
        df = pd.concat([df, expanded[result_cols]], axis=1)

        payload = json.loads(df.to_json(orient="records"))
        return json.dumps({"state_update": {"validate_results": payload}})

    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error: {str(e)}"})

