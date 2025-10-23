import numpy as np
class BlackScholesPDE:
    def __init__(self, S_max, K, T, r, sigma, option_type):
        self.S_max = S_max
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type

    def solve(self, S_points, T_points):
        S = np.linspace(0, self.S_max, S_points)
        dt = self.T / T_points
        dS = self.S_max / S_points
        V = np.zeros((S_points, T_points + 1))
        if self.option_type == 'call':
            V[:, -1] = np.maximum(0, S - self.K)
        elif self.option_type == 'put':
            V[:, -1] = np.maximum(0, self.K - S)
        alpha = 0.5 * dt * (self.sigma**2 * (np.arange(S_points) ** 2) / (dS ** 2) - self.r / dS)
        beta = -dt * (self.sigma**2 * (np.arange(S_points) ** 2) / (dS ** 2) + self.r)
        gamma = 0.5 * dt * (self.sigma**2 * (np.arange(S_points) ** 2) / (dS ** 2) + self.r / dS)
        for j in range(T_points - 1, -1, -1):
            for i in range(1, S_points - 1):
                V[i, j] = alpha[i] * V[i - 1, j + 1] + (1 - beta[i]) * V[i, j + 1] + gamma[i] * V[i + 1, j + 1]
        return V[int(S_points / 2), 0]

class PDEConvergenceTest:
    def __init__(self, pde_model):
        self.pde_model = pde_model

    def test(self):
        prices = []
        grid_points = range(100, 501, 100)
        for S_points in grid_points:
            T_points = S_points
            price = self.pde_model.solve(S_points, T_points)
            prices.append(price)
        for i in range(1, len(prices)):
            if abs(prices[i] - prices[i - 1]) > 1e-5:
                print(f"Prices do not converge at grid points {grid_points[i-1]} and {grid_points[i]}")
                return
        print("Prices converge across the grid points.")
