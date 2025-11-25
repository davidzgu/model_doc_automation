import json
import pandas as pd
from typing import List, Dict, Any

JSON_STR = List[Dict[str, Any]]

def load_json_as_df(csv_data):
    if isinstance(csv_data, str):
        data = json.loads(csv_data)
    elif isinstance(csv_data, list):
        data = csv_data
    elif isinstance(csv_data, dict):
        data = csv_data
    else:
        return False
    df = pd.DataFrame(data)
    return df
