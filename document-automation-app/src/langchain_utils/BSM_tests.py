from langchain_utils.pricing_model import black_scholes
import numpy as np
from scipy.stats import norm
from pydantic import BaseModel, Field

class OptionData(BaseModel):
    option_type: str = Field(..., descrption="'call' for call option, 'put' for put option")
    S: float = Field(..., description="underlying stock spot price")
    K: float = Field(..., description="option strike price")
    T: float = Field(..., description="time to expiration in years")
    r: float = Field(..., description="annualized risk-free interest rate")
    sigma: float = Field(..., description="annualized volatility of the underlying stock")

class TestOutput(BaseModel):
    test_name: str = Field(..., descrption="The name of this test")
    test_result: bool = Field(..., descrption="Whether test passed or not")



def test_gamma_positivity(parmeters: OptionData) -> TestOutput:
    """
    Test the gamma positivity of the Black-Scholes model. Performs the test to make sure the vanilla option priver gives positive Gamma under secific contractual conditoins. 
    This is done by bumping prices up and down by 1% and checking (priceUp+priceDown-2basePrice) staying positive. Return True if gamma positivity holds, False otherwise.
    """
    if parmeters.option_type not in ['call', 'put']:
        raise ValueError("Wrong input for argument option_type, should be either 'call' or 'put'")

    # Calculate the base price
    base_price = black_scholes(parmeters.option_type, parmeters.S, parmeters.K, parmeters.T, parmeters.r, parmeters.sigma)

    # Bump prices up and down by 1%
    bump = parmeters.S * 0.01
    price_up = black_scholes(parmeters.option_type, parmeters.S + bump, parmeters.K, parmeters.T, parmeters.r, parmeters.sigma)
    price_down = black_scholes(parmeters.option_type, parmeters.S - bump, parmeters.K, parmeters.T, parmeters.r, parmeters.sigma)

    # Check gamma positivity condition
    gamma_condition = price_up + price_down - 2 * base_price
    test_result = (gamma_condition > 0)
    return {'test_name':'gamma_positive_test', 'test_result': test_result}

def calculate_greeks(option_type, S, K, T, r, sigma):
    """
    Calculate the option price and Greeks using the Black-Scholes model.

    Parameters:
    option_type (str): 'call' for call option, 'put' for put option
    S (float): current stock price
    K (float): option strike price
    T (float): time to expiration in years
    r (float): risk-free interest rate (annualized)
    sigma (float): volatility of the underlying stock (annualized)

    Returns:
    dict: option price and Greeks
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'rho': rho,
        'theta': theta
    }

def sensitivity_test(parmeters:OptionData):
    """
    Perform sensitivity test by sliding the spot price.

    Parameters:
    option_type (str): 'call' for call option, 'put' for put option
    S (float): current stock price
    K (float): option strike price
    T (float): time to expiration in years
    r (float): risk-free interest rate (annualized)
    sigma (float): volatility of the underlying stock (annualized)
    """
    results = []
    spot_changes = np.arange(-0.025, 0.026, 0.005)  # From -2.5% to 2.5% in steps of 0.5%

    for change in spot_changes:
        new_S = parmeters.S * (1 + change)
        greeks = calculate_greeks(parmeters.option_type, new_S, parmeters.K, parmeters.T, parmeters.r, parmeters.sigma)
        results.append({
            'spot_change': change,
            'price': greeks['price'],
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'vega': greeks['vega'],
            'rho': greeks['rho'],
            'theta': greeks['theta']
        })

def digital_option_price(option_type, S, K, T, r, sigma):
    """
    Calculate the price of a digital option using the Black-Scholes model.

    Parameters:
    option_type (str): 'call' for digital call option, 'put' for digital put option
    S (float): current stock price
    K (float): option strike price
    T (float): time to expiration in years
    r (float): risk-free interest rate (annualized)
    sigma (float): volatility of the underlying stock (annualized)

    Returns:
    float: price of the digital option
    """
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        price = np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

def digital_options_test(parmeters:OptionData):
    """
    Test the consistency of the digital option pricer using the same option info
    """
    digital_call_price = digital_option_price('call', parmeters.S, parmeters.K, parmeters.T, parmeters.r, parmeters.sigma)
    digital_put_price = digital_option_price('put', parmeters.S, parmeters.K, parmeters.T, parmeters.r, parmeters.sigma)

    print(f"Digital Call Option Price: {digital_call_price:.4f}")
    print(f"Digital Put Option Price: {digital_put_price:.4f}")

    # Check consistency
    if digital_call_price + digital_put_price > 1:
        print("Inconsistency detected: The sum of digital call and put prices exceeds 1.")
        test_result = False
    else:
        print("Consistency check passed: The sum of digital call and put prices is valid.")
        test_result = True
    return {'test_name':"digital_option_test", 'test_result': test_result}
