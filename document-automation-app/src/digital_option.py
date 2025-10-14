import numpy as np
from scipy.stats import norm
class DigitalOption:
    def __init__(self, option_type, S, K, T, r, sigma):
        self.option_type = option_type
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def price(self):
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        if self.option_type == 'call':
            return np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == 'put':
            return np.exp(-self.r * self.T) * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

class DigitalOptionsTest:
    def __init__(self, digital_call, digital_put):
        self.digital_call = digital_call
        self.digital_put = digital_put

    def test(self):
        digital_call_price = self.digital_call.price()
        digital_put_price = self.digital_put.price()
        print(f"Digital Call Option Price: {digital_call_price:.4f}")
        print(f"Digital Put Option Price: {digital_put_price:.4f}")
        if digital_call_price + digital_put_price > 1:
            print("Inconsistency detected: The sum of digital call and put prices exceeds 1.")
        else:
            print("Consistency check passed: The sum of digital call and put prices is valid.")
