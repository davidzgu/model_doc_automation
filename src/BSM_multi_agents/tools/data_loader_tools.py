from langchain.tools import tool
import pandas as pd
import json
from typing import Union, Dict, Any
import numpy as np
from scipy.stats import norm
from .tool_registry import register_tool

@register_tool(tags=["io","csv"], roles=["data_loader"])
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