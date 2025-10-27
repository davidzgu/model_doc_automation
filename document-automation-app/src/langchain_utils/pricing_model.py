import numpy as np
from scipy.stats import norm

def black_scholes(option_type:str, S:float, K:float, T:float, r:float, sigma:float) -> float:
    """
    Calculate the Black-Scholes option price.

    Parameters:
    option_type (str): 'call' for call option, 'put' for put option
    S (float): current stock price
    K (float): option strike price
    T (float): time to expiration in years
    r (float): risk-free interest rate (annualized)
    sigma (float): volatility of the underlying stock (annualized)

    Returns:
    float: option price
    """
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate call and put option prices
    if option_type == 'call':
        option_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        option_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return option_price
