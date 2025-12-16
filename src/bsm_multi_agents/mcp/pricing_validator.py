from typing import Union, Dict, Any, List, Annotated
import json
import pandas as pd
import os


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
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", "_validate_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        df.to_csv(output_path, index=False)
        
        return os.path.abspath(output_path)

    except Exception as e:
        return json.dumps({"errors": [f"Error: {str(e)}"]})


def gamma_positivity_test(
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
        S = self.greeks_calculator.bs_model.S
        K = self.greeks_calculator.bs_model.K
        T = self.greeks_calculator.bs_model.T
        r = self.greeks_calculator.bs_model.r
        sigma = self.greeks_calculator.bs_model.sigma
        option_type = self.greeks_calculator.bs_model.option_type
        initial_greeks = self.greeks_calculator.calculate()
        initial_price = initial_greeks['price']
        delta = initial_greeks['delta']
        gamma = initial_greeks['gamma']
        greek_based_pnl = delta * price_change + 0.5 * gamma * (price_change ** 2)
        new_S = S + price_change
        new_price = self.greeks_calculator.calculate(new_S)['price']
        revaluation_based_pnl = new_price - initial_price
        print(f"Initial Price: {initial_price:.2f}")
        print(f"New Price: {new_price:.2f}")
        print(f"Greek-based PnL: {greek_based_pnl:.2f}")
        print(f"Revaluation-based PnL: {revaluation_based_pnl:.2f}")
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_path).replace(".csv", "_validate_results.csv")
        output_path = os.path.join(output_dir, filename)
        
        df.to_csv(output_path, index=False)
        
        return os.path.abspath(output_path)

    except Exception as e:
        return json.dumps({"errors": [f"Error: {str(e)}"]})
