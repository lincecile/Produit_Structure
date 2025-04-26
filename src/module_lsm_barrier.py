# =====================
# module_lsm.py
# =====================
import numpy as np
from typing import Tuple
from module_option import Option
from module_market import MarketEnvironment
from module_brownian import Brownian
from module_regression import RegressionEstimator

class LSM:
    def __init__(self, option: Option):
        self.option = option

    def generate_paths(self, brownian: Brownian, market: MarketEnvironment, antithetic: bool = False):
        W = brownian.Vecteur()
        if antithetic:
            W = np.vstack((W, -W))

        dt = brownian.step
        T = brownian.nb_step
        paths = np.zeros((W.shape[0], T + 1))
        paths[:, 0] = market.spot

        for t in range(1, T + 1):
            paths[:, t] = paths[:, t - 1] * np.exp((market.rate - 0.5 * market.vol ** 2) * dt + market.vol * (W[:, t - 1]))
            if market.has_dividend(t):
                paths[:, t] -= market.get_dividend(t)
        return paths

    def compute_payoff(self, paths: np.ndarray):
        return self.option.payoff_array(paths)

    def lsm_algorithm(self, paths: np.ndarray, market: MarketEnvironment, poly_degree=2, model_type="Polynomial"):
        dt = market.vol
        T = paths.shape[1] - 1
        r = market.rate
        disc = np.exp(-r * dt)

        payoffs = self.option.payoff_array(paths)
        for t in range(T - 1, 0, -1):
            spot_t = paths[:, t]
            exercise_value = self.option.payoff_array(paths[:, t: t + 1])
            itm = exercise_value > 0

            if np.any(itm):
                X = spot_t[itm]
                Y = payoffs[itm] * disc
                continuation = RegressionEstimator(X, Y, degree=poly_degree, model_type=model_type).Regression()
                exercise = exercise_value[itm] > continuation
                payoffs[itm] = np.where(exercise, exercise_value[itm], payoffs[itm] * disc)
            else:
                payoffs *= disc
        return payoffs

    def price(self, brownian: Brownian, market: MarketEnvironment, poly_degree=2, model_type="Polynomial", antithetic=False) -> Tuple[float, float, Tuple[float, float]]:
        paths = self.generate_paths(brownian, market, antithetic)
        if not self.option.americaine:
            payoffs = self.option.payoff_array(paths) * np.exp(-market.rate * self.option.maturity)
        else:
            payoffs = self.lsm_algorithm(paths, market, poly_degree, model_type)

        prix = np.mean(payoffs)
        std = np.std(payoffs) / np.sqrt(len(payoffs))
        ic = (prix - 2 * std, prix + 2 * std)
        return prix, std, ic
