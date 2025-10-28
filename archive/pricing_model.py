""" 
Action item:
@Ricky & @Vivian to generate the below:
    1. Simple Equity BSM Model 
    2. Find public input data for the model 
"""

class Equity_BSM_Model:
    def __init__(self, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility):
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

    def calculate_option_price(self):
        # Placeholder for actual BSM option pricing logic
        pass

    def calculate_greeks(self):
        # Placeholder for actual Greeks calculation logic
        pass