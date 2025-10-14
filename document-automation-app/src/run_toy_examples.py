from black_scholes import BlackScholes
from gamma_positivity_test import GammaPositivityTest
from black_scholes_pde import BlackScholesPDE, PDEConvergenceTest
from greeks_calculator import GreeksCalculator, SensitivityTest
from pnl_explain_test import PnLExplainTest
from digital_option import DigitalOption, DigitalOptionsTest

# Toy parameters
S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

# Black-Scholes pricing
bs_call = BlackScholes('call', S, K, T, r, sigma)
bs_put = BlackScholes('put', S, K, T, r, sigma)
print(f"Call Option Price: {bs_call.price():.2f}")
print(f"Put Option Price: {bs_put.price():.2f}")

# Gamma Positivity Test
gamma_test_call = GammaPositivityTest(bs_call)
gamma_test_put = GammaPositivityTest(bs_put)
print(f"Gamma Positivity Test for Call Option: {'Positive' if gamma_test_call.test() else 'Negative'}")
print(f"Gamma Positivity Test for Put Option: {'Positive' if gamma_test_put.test() else 'Negative'}")

# PDE Convergence Test
S_max = 200
pde_model = BlackScholesPDE(S_max, K, T, r, sigma, 'call')
pde_test = PDEConvergenceTest(pde_model)
pde_test.test()

# Greeks and Sensitivity Test
greeks_calc = GreeksCalculator(bs_call)
sensitivity_test = SensitivityTest(greeks_calc)
sensitivity_results = sensitivity_test.test()
for result in sensitivity_results:
    print(f"Spot Change: {result['spot_change']:.2%}, Price: {result['price']:.2f}, "
          f"Delta: {result['delta']:.4f}, Gamma: {result['gamma']:.4f}, "
          f"Vega: {result['vega']:.4f}, Rho: {result['rho']:.4f}, Theta: {result['theta']:.4f}")

# PnL Explain Test
pnl_test = PnLExplainTest(greeks_calc)
price_change = 1.0
pnl_test.test(price_change)

# Digital Option Test
digital_call = DigitalOption('call', S, K, T, r, sigma)
digital_put = DigitalOption('put', S, K, T, r, sigma)
digital_test = DigitalOptionsTest(digital_call, digital_put)
digital_test.test()
