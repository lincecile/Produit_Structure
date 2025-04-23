# hestonpricer/models/heston_parameters.py

class HestonParameters:
    def __init__(self, kappa, theta, v0, sigma, rho, lmbda=0):
        """
        Heston model parameters
        
        Args:
            kappa: Mean reversion speed
            theta: Long-term variance
            v0: Initial variance
            sigma: Volatility of volatility
            rho: Correlation between asset returns and volatility
            lmbda: Volatility risk premium (default: 0)
        """
        self.kappa = kappa
        self.theta = theta
        self.v0 = v0
        self.sigma = sigma
        self.rho = rho
        self.lmbda = lmbda
