import pandas as pd
import numpy as np
from datetime import datetime
import os

def simulate_equity_options(n_pairs=25, output_path="data/input/simulated_equity_options.csv"):
    """
    Simulates equity option data with Call-Put pairs for testing.
    
    Args:
        n_pairs (int): Number of Call-Put pairs to generate. Total rows will be n_pairs * 2.
        output_path (str): Path to save the generated CSV.
    """
    # Define underlying stock info (Realistic Spot and Volatility range)
    underlyings_info = {
        'AAPL': {'S': 230.0, 'sigma': 0.25},
        'MSFT': {'S': 410.0, 'sigma': 0.22},
        'GOOGL': {'S': 180.0, 'sigma': 0.28},
        'TSLA': {'S': 250.0, 'sigma': 0.45},
        'NVDA': {'S': 130.0, 'sigma': 0.40}
    }
    
    tickers = list(underlyings_info.keys())
    data = []
    
    # Common parameters
    today = datetime.now().strftime("%Y-%m-%d")
    risk_free_rate = 0.045 # 4.5%
    
    for i in range(n_pairs):
        ticker = np.random.choice(tickers)
        base_S = underlyings_info[ticker]['S']
        base_sigma = underlyings_info[ticker]['sigma']
        
        # Randomize parameters slightly around base
        S = base_S * np.random.uniform(0.95, 1.05)
        # Randomize K to be around S (ITM, ATM, OTM)
        K = S * np.random.uniform(0.8, 1.2)
        T = np.random.uniform(0.1, 2.0) # 1 month to 2 years
        sigma = base_sigma * np.random.uniform(0.9, 1.1)
        
        # Create a Call-Put pair
        row_call = {
            'ID': i * 2 + 1,
            'date': today,
            'underlying': ticker,
            'S': round(S, 2),
            'K': round(K, 2),
            'T': round(T, 3),
            'r': risk_free_rate,
            'sigma': round(sigma, 3),
            'option_type': 'call',
            'style': 'EURO',
            'asset_class': 'Equity'
        }
        
        row_put = row_call.copy()
        row_put['ID'] = i * 2 + 2
        row_put['option_type'] = 'put'
        
        data.append(row_call)
        data.append(row_put)
    
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} rows of equity option data at {output_path}")
    return output_path

if __name__ == "__main__":
    simulate_equity_options()
