class GammaPositivityTest:
    def __init__(self, bs_model):
        self.bs_model = bs_model

    def test(self):
        S = self.bs_model.S
        K = self.bs_model.K
        T = self.bs_model.T
        r = self.bs_model.r
        sigma = self.bs_model.sigma
        option_type = self.bs_model.option_type
        base_price = self.bs_model.price()
        bump = S * 0.01
        from black_scholes import BlackScholes
        price_up = BlackScholes(option_type, S + bump, K, T, r, sigma).price()
        price_down = BlackScholes(option_type, S - bump, K, T, r, sigma).price()
        gamma_condition = price_up + price_down - 2 * base_price
        return gamma_condition > 0
