from langchain.tools import tool
import numpy as np
import json
import pandas as pd

from src.core.greeks_calculator import greeks_calculator



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



# @tool
# def sensitivity_test(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> str:
#     """
#     Runs a sensitivity test by perturbing the spot price around S and computing Greeks at each perturbed spot.

#     Args:
#         option_type (str): 'call' or 'put'
#         S (float): Base spot price
#         K (float): Strike price
#         T (float): Time to expiration in years
#         r (float): Risk-free rate
#         sigma (float): Volatility

#     Returns:
#         str: JSON list of sensitivity results, each entry contains spot_change and greeks
#     """
#     try:
#         print(f"Testing: Running sensitivity test for {option_type} with base S={S}, K={K}, T={T}, r={r}, sigma={sigma}")

#         spot_changes = np.arange(-0.025, 0.026, 0.005)
#         results = []

#         for change in spot_changes:
#             new_S = S * (1 + float(change))
#             # Call the greeks_calculator tool via its .run API with a single dict payload
#             try:
#                 resp = greeks_calculator.run({
#                     "option_type": option_type,
#                     "S": float(new_S),
#                     "K": float(K),
#                     "T": float(T),
#                     "r": float(r),
#                     "sigma": float(sigma),
#                 })
#             except TypeError:
#                 # Fallback: some LangChain versions may allow calling the tool object directly
#                 resp = greeks_calculator(option_type, new_S, K, T, r, sigma)

#             try:
#                 greeks = json.loads(resp)
#             except Exception:
#                 greeks = {"error": "failed to parse greeks response", "raw": resp}

#             entry = {"spot_change": float(change)}
#             if isinstance(greeks, dict) and "error" in greeks:
#                 entry["error"] = greeks.get("error")
#             else:
#                 # copy expected fields
#                 for k in ("price", "delta", "gamma", "vega", "rho", "theta"):
#                     entry[k] = greeks.get(k)

#             results.append(entry)

#         return json.dumps(results)
#     except Exception as e:
#         return json.dumps({"error": str(e)})
