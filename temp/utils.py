import pandas as pd
from io import StringIO
from typing import List, Dict, Any, Union

JSON_STR = List[Dict[str, Any]]

def load_json_as_df(json_data: Union[str, List[Dict], Dict]):
    if isinstance(json_data, str):
        return pd.read_json(StringIO(json_data), orient='records')
    elif isinstance(json_data, (list, dict)):
        return pd.DataFrame(json_data)
    return False
