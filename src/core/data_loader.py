
from langchain.tools import tool
import pandas as pd



@tool
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