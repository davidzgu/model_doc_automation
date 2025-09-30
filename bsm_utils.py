import numpy as np
from scipy.stats import norm
from langchain.tools import tool
import pandas as pd


class BSMUtils:
    @tool
    def black_scholes(option_type, S, K, T, r, sigma) -> str:
        """
        Calculate the Black-Scholes option price.
        """
        S = float(S)
        K = float(K)
        T = float(T)
        r = float(r)
        sigma = float(sigma)
        option_type = option_type.lower()

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            option_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        elif option_type == 'put':
            option_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        return str(option_price)

    @tool
    def read_csv_tool(filepath: str, nrows: int = 5) -> str:
        """
        Reads a CSV file and returns the first nrows as a string.
        """
        try:
            df = pd.read_csv(filepath)
            return df.head(nrows).to_string(index=False)
        except Exception as e:
            return f"Error reading CSV: {e}"