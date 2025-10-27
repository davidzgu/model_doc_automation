import numpy as np
from scipy.stats import norm
from langchain.tools import tool
import pandas as pd


from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
import json
from typing import Type, Optional, Union, Dict, Any


@tool
def bsm_calculator(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
    """
    Calculates the Black-Scholes price for European call or put options.
    This function implements the Black-Scholes formula to compute the theoretical price of a European option
    (either 'call' or 'put') given the current stock price, strike price, time to expiration, risk-free interest rate,
    and volatility of the underlying asset.
    Args:
        option_type (str): Type of the option, either 'call' or 'put'.
        S (float): Current price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized standard deviation).
    Returns:
        str: The calculated option price as a string.
    Raises:
        ValueError: If option_type is not 'call' or 'put'.
    # AI Agent Explanation:
    # This function uses the Black-Scholes mathematical model to determine the fair price of a European option.
    # It computes intermediate variables (d1 and d2) and applies the formula for either a call or put option.
    # The result is returned as a string for further processing or display.
    """
    #print(f"Testing: Calculating Black-Scholes price for a {option_type} option with S={S}, K={K}, T={T}, r={r}, sigma={sigma}")

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
def csv_loader(filepath: str) -> str:
    """
    Reads a CSV file from the specified filepath and returns the first five rows in JSON format without the index. If an error occurs during reading,
    returns an error message.

    Args:
        filepath (str): The path to the CSV file to be read.

    Returns:
        str: JSON string of the first five rows of the CSV file, or an error message.
    """

    try:
        df = pd.read_csv(filepath)
        return df.head(5).to_json(index=False)
    except Exception as e:
        return f"Error reading CSV: {e}"


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


@tool
def sensitivity_test_new(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
    """
        Runs a sensitivity test by perturbing the spot price around S and computing Greeks at each perturbed spot.

    Args:
        option_type (str): 'call' or 'put'
        S (float): Base spot price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free rate
        sigma (float): Volatility

    Returns:
        str: JSON list of sensitivity results, each entry contains spot_change and greeks
    """
    spot_changes = np.arange(-0.025, 0.026, 0.005)
    results = []

    for change in spot_changes:
        new_S = S * (1 + float(change))
        # Call greeks_calculator as a tool via .run with a single dict payload
        resp = greeks_calculator.run({
            "option_type": option_type,
            "S": float(new_S),
            "K": float(K),
            "T": float(T),
            "r": float(r),
            "sigma": float(sigma),
        })

        greeks = json.loads(resp)

        entry = {"spot_change": float(change)}
        for k in ("price", "delta", "gamma", "vega", "rho", "theta"):
            entry[k] = greeks.get(k)

        results.append(entry)

    # Build a pandas DataFrame table for convenience
    table_df = pd.DataFrame(results)

    # Return both the raw results list and a table representation (as list of records)
    return json.dumps({
        "results": results,
        "table": table_df.to_dict(orient="records")
    })


@tool
def sensitivity_test(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
    """
    Runs a sensitivity test by perturbing the spot price around S and computing Greeks at each perturbed spot.

    Args:
        option_type (str): 'call' or 'put'
        S (float): Base spot price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free rate
        sigma (float): Volatility

    Returns:
        str: JSON list of sensitivity results, each entry contains spot_change and greeks
    """
    try:
        print(f"Testing: Running sensitivity test for {option_type} with base S={S}, K={K}, T={T}, r={r}, sigma={sigma}")

        spot_changes = np.arange(-0.025, 0.026, 0.005)
        results = []

        for change in spot_changes:
            new_S = S * (1 + float(change))
            # Call the greeks_calculator tool via its .run API with a single dict payload
            try:
                resp = greeks_calculator.run({
                    "option_type": option_type,
                    "S": float(new_S),
                    "K": float(K),
                    "T": float(T),
                    "r": float(r),
                    "sigma": float(sigma),
                })
            except TypeError:
                # Fallback: some LangChain versions may allow calling the tool object directly
                resp = greeks_calculator(option_type, new_S, K, T, r, sigma)

            try:
                greeks = json.loads(resp)
            except Exception:
                greeks = {"error": "failed to parse greeks response", "raw": resp}

            entry = {"spot_change": float(change)}
            if isinstance(greeks, dict) and "error" in greeks:
                entry["error"] = greeks.get("error")
            else:
                # copy expected fields
                for k in ("price", "delta", "gamma", "vega", "rho", "theta"):
                    entry[k] = greeks.get(k)

            results.append(entry)

        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})




# # Define proper input schema
# class OptionPricingInput(BaseModel):
#     option_type: str = Field(description="Type of option: 'call' or 'put'")
#     S: float = Field(description="Current price of the underlying asset")
#     K: float = Field(description="Strike price of the option")
#     T: float = Field(description="Time to expiration in years")
#     r: float = Field(description="Risk-free interest rate")
#     sigma: float = Field(description="Volatility")

#     class Config:
#         arbitrary_types_allowed = True  # Allow unsupported types

# @tool
# class black_scholes(BaseTool):
#     name: str = "Black-Scholes Calculator"
#     description: str = "Calculates the Black-Scholes price for European call or put options."
#     args_schema: Type[BaseModel] = OptionPricingInput

#     def _run(self, option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
#         """
#         Calculates the Black-Scholes price for European call or put options.
#         This function implements the Black-Scholes formula to compute the theoretical price of a European option
#         (either 'call' or 'put') given the current stock price, strike price, time to expiration, risk-free interest rate,
#         and volatility of the underlying asset.
#         Args:
#             option_type (str): Type of the option, either 'call' or 'put'.
#             S (float): Current price of the underlying asset.
#             K (float): Strike price of the option.
#             T (float): Time to expiration in years.
#             r (float): Risk-free interest rate (annualized).
#             sigma (float): Volatility of the underlying asset (annualized standard deviation).
#         Returns:
#             str: The calculated option price as a string.
#         Raises:
#             ValueError: If option_type is not 'call' or 'put'.
#         # AI Agent Explanation:
#         # This function uses the Black-Scholes mathematical model to determine the fair price of a European option.
#         # It computes intermediate variables (d1 and d2) and applies the formula for either a call or put option.
#         # The result is returned as a string for further processing or display.
#         """
#         print(f"Testing: Calculating Black-Scholes price for a {option_type} option with S={S}, K={K}, T={T}, r={r}, sigma={sigma}")

#         option_type = option_type.lower()

#         d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#         d2 = d1 - sigma * np.sqrt(T)

#         if option_type == 'call':
#             option_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
#         elif option_type == 'put':
#             option_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
#         else:
#             raise ValueError("option_type must be 'call' or 'put'")
#         return str(option_price)



# class ReadCSVInput(BaseModel):
#     filepath: str = Field(description="The path to the CSV file to be read.")
    
#     class Config:
#         arbitrary_types_allowed = True  # Allow unsupported types

# @tool
# class read_csv_tool(BaseTool):
#     name: str = "CSV Reader"
#     description: str = "Reads a CSV file and returns the first five rows in JSON format."
#     # AI Agent Comment:
#     # This function allows an AI agent to preview the contents of a CSV file.
#     # It loads the file into a pandas DataFrame, extracts the top five rows, and converts them
#     # to JSON for display or further processing. If the file cannot be read, it provides
#     # an error message for debugging or user feedback.
#     args_schema: Type[BaseModel] = ReadCSVInput
 
#     def _run(self, filepath: str) -> str:
#         """
#         Reads a CSV file from the specified filepath and returns the first five rows in JSON format without the index. If an error occurs during reading,
#         returns an error message.

#         Args:
#             filepath (str): The path to the CSV file to be read.

#         Returns:
#             str: JSON string of the first five rows of the CSV file, or an error message.
#         """

#         try:
#             df = pd.read_csv(filepath)
#             return df.head(5).to_json(index=False)
#         except Exception as e:
#             return f"Error reading CSV: {e}"


@tool
def batch_bsm_calculator(csv_data: Union[str, Dict[str, Any]]) -> str:
    """
    Batch calculates Black-Scholes option prices for multiple options from CSV data.
    This tool is more efficient than calling bsm_calculator multiple times.

    Args:
        csv_data (str): JSON string or dict of CSV data with columns: option_type, S, K, T, r, sigma
                       (as returned by csv_loader tool)

    Returns:
        str: Markdown table with input parameters and calculated option prices for each row

    Example:
        Input CSV data should have columns: option_type, S, K, T, r, sigma
        Returns a formatted markdown table with all inputs and calculated prices
    """
    try:
        # Parse the JSON data (handle both string and dict)
        if isinstance(csv_data, str):
            data = json.loads(csv_data)
        elif isinstance(csv_data, dict):
            data = csv_data
        else:
            return f"Error: csv_data must be a string or dict, got {type(csv_data)}"

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data)

        # Required columns
        required_cols = ['option_type', 'S', 'K', 'T', 'r', 'sigma']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Error: Missing required columns: {missing_cols}. Available columns: {list(df.columns)}"

        # Calculate option price for each row
        prices = []
        for idx, row in df.iterrows():
            option_type = str(row['option_type']).lower()
            S = float(row['S'])
            K = float(row['K'])
            T = float(row['T'])
            r = float(row['r'])
            sigma = float(row['sigma'])

            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            # Calculate option price
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type == 'put':
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                price = None

            prices.append(price)

        # Add prices to DataFrame
        df['BSM_Price'] = prices

        # Format as markdown table
        result = "## Black-Scholes Option Pricing Results\n\n"
        result += df.to_markdown(index=False)

        return result

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format. {str(e)}"
    except Exception as e:
        return f"Error calculating option prices: {str(e)}"