import numpy as np
from scipy.stats import norm
class GreeksCalculator:
    def __init__(self, bs_model):
        self.bs_model = bs_model

    def calculate(self, S=None):
        if S is None:
            S = self.bs_model.S
        K = self.bs_model.K
        T = self.bs_model.T
        r = self.bs_model.r
        sigma = self.bs_model.sigma
        option_type = self.bs_model.option_type
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
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

class SensitivityTest:
    def __init__(self, greeks_calculator):
        self.greeks_calculator = greeks_calculator

    def test(self):
        S = self.greeks_calculator.bs_model.S
        results = []
        spot_changes = np.arange(-0.025, 0.026, 0.005)
        for change in spot_changes:
            new_S = S * (1 + change)
            greeks = self.greeks_calculator.calculate(new_S)
            results.append({
                'spot_change': change,
                'price': greeks['price'],
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'vega': greeks['vega'],
                'rho': greeks['rho'],
                'theta': greeks['theta']
            })
        return results
