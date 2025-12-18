from typing import Union, Dict, Any, List, Annotated
import json
import pandas as pd
import os
import numpy as np
from scipy.stats import norm


def _validate_greeks_rules(
        option_type: str,
        price: float, 
        delta: float, 
        gamma: float, 
        vega: float
    ) -> str:
    """
    Validate Greeks against business rules.
    Returns list of validation results.
    """
    validations_result = "passed"
    validations_details = []

    # Rule 1: price > 0
    try:
        price_valid = price > 0
        if not price_valid:
            validations_result = "failed"
            validations_details.append(f"Price {price} is not positive")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Price Error: {str(e)}")

    # Rule 2: delta range
    try:
        if option_type.lower() == 'call':
            delta_valid = 0 <= delta <= 1
            expected_range = "[0, 1]"
        else:
            delta_valid = -1 <= delta <= 0
            expected_range = "[-1, 0]"
        if not delta_valid:
            validations_result = "failed"
            validations_details.append(f"Delta {delta:.4f} outside {expected_range}")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Delta Error: {str(e)}")

    # Rule 3: gamma >= 0
    try:
        gamma_valid = gamma >= 0
        if not gamma_valid:
            validations_result = "failed"
            validations_details.append(f"Gamma {gamma:.4f} is negative")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Gamma Error: {str(e)}")

    # Rule 4: vega >= 0
    try:
        vega_valid = vega >= 0
        if not gamma_valid:
            validations_result = "failed"
            validations_details.append(f"Vega {vega:.4f} is negative")
    except Exception as e:
        validations_result = "failed"
        validations_details.append(f"Vega Error: {str(e)}")

    results = {
        "validations_result": validations_result,
        "validations_details": validations_details
    }
    return results

def _black_scholes(option_type, S, K, T, r, sigma):
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

def test_gamma_positivity(option_type, S, K, T, r, sigma):
    """
    Test the gamma positivity of the Black-Scholes model.

    Parameters:
    option_type (str): 'call' for call option, 'put' for put option
    S (float): current stock price
    K (float): option strike price
    T (float): time to expiration in years
    r (float): risk-free interest rate (annualized)
    sigma (float): volatility of the underlying stock (annualized)

    Returns:
    bool: True if gamma positivity holds, False otherwise
    """
    # Calculate the base price
    base_price = _black_scholes(option_type, S, K, T, r, sigma)

    # Bump prices up and down by 1%
    bump = S * 0.01
    price_up = _black_scholes(option_type, S + bump, K, T, r, sigma)
    price_down = _black_scholes(option_type, S - bump, K, T, r, sigma)

    # Check gamma positivity condition
    gamma_condition = price_up + price_down - 2 * base_price
    return gamma_condition > 0

def validate_greeks_to_file(
    input_path: str, output_dir: str
) -> str:
    """
    Validate Greeks for ALL options from CSV data.

    For each option:
    - Validates: price > 0
    - Validates: delta in [0,1] for calls, [-1,0] for puts
    - Validates: gamma >= 0, vega >= 0

    Args:
        state: InjectedState, state from the workflow, which contains csv_data


    Returns:
        JSON string containing validate_results
    """
    try:
        if not os.path.exists(input_path):
            return f"Error: Input file not found at {input_path}"

        df = pd.read_csv(input_path)
        required_cols = ['option_type', 'price', 'delta', 'gamma', 'vega']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps({"errors": [f"Missing columns: {missing}"]})

        def calc_row(row):
            res = _validate_greeks_rules(
                row['option_type'], 
                row['price'], 
                row['delta'], 
                row['gamma'], 
                row['vega'], 
            )
            return res
        
        expanded = df.apply(calc_row, axis=1).apply(pd.Series)
        result_cols = ['validations_result','validations_details']
        for col in result_cols:
            if col not in expanded:
                expanded[col] = pd.NA
        df = pd.concat([df, expanded[result_cols]], axis=1)

        # New column
        df["gamma_positive"] = False

        for idx, row in df.iterrows():
            option_type = row["option_type"]
            S = row["S"]
            K = row["K"]
            T = row["T"]
            r = row["r"]
            sigma = row["sigma"]

            # Run your test
            gamma_pos = test_gamma_positivity(option_type, S, K, T, r, sigma)

            # Save result
            df.at[idx, "gamma_positive"] = gamma_pos

        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", "_validate_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        df.to_csv(output_path, index=False)
        
        return os.path.abspath(output_path)

    except Exception as e:
        return json.dumps({"errors": [f"Error: {str(e)}"]})


