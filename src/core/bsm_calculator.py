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


