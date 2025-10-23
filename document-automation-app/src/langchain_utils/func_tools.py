import pandas as pd
from pydantic import BaseModel, Field

class CSVReadInput(BaseModel):
    csv_path: str = Field(..., description="Path to the CSV file containing option data")
    n_rows: None|int = Field(None, description="Number of rows to read from the CSV. If None, read all data")

def read_csv(input_data: CSVReadInput) -> dict:
    """
    Reads a CSV file and returns a JSON preview.
    Use this tool to understand the structure of the data before running tests.
    """
    df = pd.read_csv(input_data.csv_path, nrows=input_data.n_rows)
    output = {
        "columns": df.columns.tolist(),
        "preview": df.to_dict(orient="records"),
        "n_rows": len(df),
    }
    return output


