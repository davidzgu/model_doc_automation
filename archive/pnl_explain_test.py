class PnLExplainTest:
    def __init__(self, greeks_calculator):
        self.greeks_calculator = greeks_calculator

    def test(self, price_change):
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
