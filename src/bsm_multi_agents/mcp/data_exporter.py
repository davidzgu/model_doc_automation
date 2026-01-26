import pandas as pd
import os
from datetime import datetime
from pathlib import Path



def save_to_local(df: pd.DataFrame, folder_name: str, prefix: str) -> str:
    base_dir = Path.cwd() / "data" / "cache" / folder_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = base_dir / f"{prefix}.csv"
    
    df.to_csv(file_path, index=False)
    return str(file_path.absolute())