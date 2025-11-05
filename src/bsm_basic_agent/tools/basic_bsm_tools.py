from langchain.tools import tool
import pandas as pd
import json
from typing import Union, Dict, Any
import numpy as np
from scipy.stats import norm
from .tool_registry import register_tool

@register_tool(tags=["io","csv"], roles=["basic_bsm_agent"])
@tool("read_csv_records")
def csv_loader(filepath: str) -> str:
    """
    Reads a CSV file from the specified filepath and returns the first five rows in JSON format without the index. If an error occurs during reading,
    returns an error message.

    Args:
        filepath (str): The path to the CSV file to be read.

    Returns:
        str: JSON string of the first five rows of the CSV file, or an error message.
    """

    try:
        df = pd.read_csv(filepath)
        return df.to_json(index=False)
    except Exception as e:
        return f"Error reading CSV: {e}"
    

@register_tool(tags=["bsm","price"], roles=["basic_bsm_agent"])
@tool("calculate_bsm_price")
def bsm_calculator(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
    """
    Calculates the Black-Scholes price for European call or put options.

    Args:
        option_type (str): Type of the option, either 'call' or 'put'
        S (float): Current price of the underlying asset
        K (float): Strike price of the option
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the underlying asset (annualized standard deviation)

    Returns:
        str: The calculated option price as a string
    """
    option_type = option_type.lower()

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        option_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return str(option_price)





@register_tool(tags=["bsm","price","batch"], roles=["calculator"])
@tool("calculate_bsm_price_batch")
def batch_bsm_calculator(csv_data: Union[str, Dict[str, Any]]) -> Dict[str,str]:
    """
    Batch calculates Black-Scholes option prices for multiple options from CSV data.

    Args:
        csv_data: JSON string or dict of CSV data with columns: option_type, S, K, T, r, sigma

    Returns:
        str: Markdown table with input parameters and calculated option prices for each row
    """
    try:
        if isinstance(csv_data, str):
            data = json.loads(csv_data)
        elif isinstance(csv_data, dict):
            data = csv_data
        else:
            return f"Error: csv_data must be a string or dict, got {type(csv_data)}"

        df = pd.DataFrame(data)

        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            result = {
                "type": "Error Information",
                "data": "Error: Missing required columns: {missing_cols}"
            }
            return result

        def calc_row(row):
            return bsm_calculator.func(
                option_type=row['option_type'], 
                S=row['S'], 
                K=row['K'], 
                T=row['T'], 
                r=row['r'], 
                sigma=row['sigma'], 
            )
        
        df['BSM_Price'] = df.apply(
            calc_row, axis=1
        )


        result = {
            "type": "Black-Scholes Option Pricing Results",
            "data": df.to_json(orient="records")
        }

        return result

    except json.JSONDecodeError as e:
        result = {
            "type": "Error Information",
            "data": f"Error: Invalid JSON format. {str(e)}"
        }
        return result
    except Exception as e:
        result = {
            "type": "Error Information",
            "data": f"Error calculating option prices: {str(e)}"
        }
        return result
