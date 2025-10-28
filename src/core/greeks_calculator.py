from langchain.tools import tool
import numpy as np
import json
from scipy.stats import norm


@tool
def greeks_calculator(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
    """
    Calculates option Greeks for a European call or put using the Black-Scholes model.

    Args:
        option_type (str): 'call' or 'put'
        S (float): Spot price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free rate (annualized)
        sigma (float): Volatility (annualized)

    Returns:
        str: JSON string containing price, delta, gamma, vega, rho, theta
    """
    try:
        #print(f"Testing: Calculating Greeks for a {option_type} option with S={S}, K={K}, T={T}, r={r}, sigma={sigma}")

        option_type = option_type.lower()

        if T <= 0 or sigma <= 0:
            return json.dumps({"error": "T and sigma must be positive"})

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = float(norm.cdf(d1))
            gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
            vega = float(S * norm.pdf(d1) * np.sqrt(T))
            rho = float(K * T * np.exp(-r * T) * norm.cdf(d2))
            theta = float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)))
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = float(norm.cdf(d1) - 1)
            gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
            vega = float(S * norm.pdf(d1) * np.sqrt(T))
            rho = float(-K * T * np.exp(-r * T) * norm.cdf(-d2))
            theta = float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)))
        else:
            return json.dumps({"error": "option_type must be 'call' or 'put'"})

        results = {
            "price": float(price),
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho": rho,
            "theta": theta,
        }

        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})