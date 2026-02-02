
import os
import pandas as pd
from bsm_multi_agents.mcp.pricing_validator import run_stress_test_to_file

# Create a sample CSV
data = {
    'option_type': ['call', 'put', 'call'],
    'S': [100.0, 100.0, 150.0],
    'K': [100.0, 100.0, 140.0],
    'T': [0.5, 0.5, 0.25],
    'r': [0.05, 0.05, 0.03],
    'sigma': [0.2, 0.2, 0.3]
}
df = pd.DataFrame(data)
os.makedirs("test_data", exist_ok=True)
csv_path = "test_data/sample_options.csv"
df.to_csv(csv_path, index=False)

# Run stress test
output_dir = "test_outputs"
result_path = run_stress_test_to_file(csv_path, output_dir)
print(f"Result Path: {result_path}")

if os.path.exists(result_path):
    result_df = pd.read_csv(result_path)
    print("\nResult Columns:")
    print(result_df.columns.tolist())
    print("\nFirst row sample:")
    print(result_df.iloc[0].to_dict())
else:
    print(f"Error: Output file not found at {result_path}")
