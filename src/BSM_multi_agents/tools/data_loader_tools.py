from langchain.tools import tool
import pandas as pd
import json
from .tool_registry import register_tool

@register_tool(tags=["io","csv"], roles=["data_loader"])
@tool("csv_loader")
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
        rows = df.to_dict(orient="records")
        return json.dumps({"state_update": {"csv_data": rows}})
    except Exception as e:
        return json.dumps({"state_update": {"errors": [f"Error reading CSV: {e}"]}})