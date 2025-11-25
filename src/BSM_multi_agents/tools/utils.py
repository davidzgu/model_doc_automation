import pandas as pd
from io import StringIO
from typing import List, Dict, Any

JSON_STR = List[Dict[str, Any]]

def load_json_as_df(json_str: JSON_STR):
    df = pd.read_json(StringIO(json_str), orient='records')
    return df
