import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

def generate_complex_data(num_options=200):
    np.random.seed(42)
    random.seed(42)

    data = []
    start_date = datetime(2025, 9, 1)

    asset_classes = ['Equity', 'Commodity', 'FX']
    option_types = ['call', 'put']

    for i in range(num_options):
        # Basic attributes
        date = start_date + timedelta(days=i % 30)
        asset_class = random.choice(asset_classes)
        opt_type = random.choice(option_types)
        
        # Base parameters depending on asset class (just for flavor)
        if asset_class == 'Equity':
            S = np.random.normal(100, 10)
            sigma = np.random.uniform(0.15, 0.40)
        elif asset_class == 'Commodity':
            S = np.random.normal(50, 5)
            sigma = np.random.uniform(0.20, 0.60)
        else: # FX
            S = np.random.normal(1.1, 0.05)
            sigma = np.random.uniform(0.05, 0.15)

        # Strike near spot
        K = S * np.random.uniform(0.9, 1.1)
        
        # Maturity
        T = np.random.uniform(0.1, 2.0)
        
        # Risk free rate
        r = np.random.uniform(0.01, 0.05)

        # Introduce Anomalies (approx 10% of data)
        anomaly_type = None
        if random.random() < 0.1:
            anomaly_choice = random.choice(['missing_sigma', 'negative_T', 'extreme_vol', 'negative_price', 'missing_r'])
            
            if anomaly_choice == 'missing_sigma':
                sigma = None # Missing value
            elif anomaly_choice == 'negative_T':
                T = -0.1 # Expired/Invalid
            elif anomaly_choice == 'extreme_vol':
                sigma = 2.5 # 250% volatility
            elif anomaly_choice == 'negative_price':
                S = -10 # Impossible price
            elif anomaly_choice == 'missing_r':
                r = None

        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'S': S,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'option_type': opt_type,
            'asset_class': asset_class
        })

    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = Path(__file__).resolve().parents[1] / "data" / "input" / "dummy_options.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} options at {output_path}")
    return output_path

if __name__ == "__main__":
    generate_complex_data()
