""" 
Action item:
@Ricky & @Vivian to generate the below:
    1. Simple Equity BSM Model 
    2. Find public input data for the model 
"""
import json
class Equity_BSM_Model:
    def __init__(self, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility):
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

    def calculate_option_price(self):
        # Placeholder for actual BSM option pricing logic
        res = {
            "script_output": 3,
            "output_desciption": "BSM pricing result"
        }
        return res

    def calculate_greeks(self):
        # Placeholder for actual Greeks calculation logic
        pass

if __name__ == "__main__":
    BS_model = Equity_BSM_Model(1, 2, 3, 4, 5)
    print(json.dumps(BS_model.calculate_option_price()))