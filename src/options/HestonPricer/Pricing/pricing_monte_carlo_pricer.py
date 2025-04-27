# hestonpricer/pricing/monte_carlo_pricer.py
import numpy as np
# from src.options.HestonPricer.Pricing.pricing_pricer_base import PricerBase
# from src.options.HestonPricer.Utils.utils_random_number_generator import RandomNumberGenerator
# from src.options.HestonPricer.Utils.utils_stats import Stats

from ..Pricing.pricing_pricer_base import PricerBase
from ..Utils.utils_random_number_generator import RandomNumberGenerator
from ..Utils.utils_stats import Stats
from multiprocessing import Pool
import functools

class MonteCarloPricer(PricerBase):
    def __init__(self, option, heston_parameters, nb_paths=100000, nb_steps=200):
        """
        Monte Carlo pricer for Heston model
        
        Args:
            option: Option object
            heston_parameters: Heston model parameters
            nb_paths: Number of Monte Carlo paths
            nb_steps: Number of time steps 
        """
        super().__init__(option, heston_parameters=heston_parameters)
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
    
    def price(self, random_seed=42):
        """
        Calculate option price using Monte Carlo simulation
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            float: Option price
        """
        np.random.seed(random_seed)
        
        T = self.option.maturity
        S0 = self.option.prix_spot
        r = self.option.risk_free_rate
        kappa = self.heston_parameters.kappa
        theta = self.heston_parameters.theta
        V0 = self.heston_parameters.v0
        sigma = self.heston_parameters.sigma
        rho = self.heston_parameters.rho
        dt = T / self.nb_steps
        
        sum_payoffs = 0.0
        
        for i in range(self.nb_paths):
            # Initialize paths for spot price and variance
            path = [np.zeros(self.nb_steps) for _ in range(2)]
            
            St = [S0, S0]
            Vt = [V0, V0]
            
            for step in range(self.nb_steps):
                # Generate correlated random variables
                sample = RandomNumberGenerator.generate_correlated_normals(rho)
                dW1, dW2 = sample[0], sample[1]
                anti_dW1, anti_dW2 = -dW1, -dW2
                
                # Ensure volatility is non-negative
                Vt[0] = max(0, Vt[0])
                Vt[1] = max(0, Vt[1])
                sqrt_Vt0 = np.sqrt(Vt[0])
                sqrt_Vt1 = np.sqrt(Vt[1])
                
                # Update variance using Milstein scheme
                Vt[0] += kappa * (theta - Vt[0]) * dt + sigma * sqrt_Vt0 * dW2 * np.sqrt(dt) + 0.25 * sigma**2 * dt * (dW2**2 - 1)
                St[0] *= np.exp((r - 0.5 * Vt[0]) * dt + sqrt_Vt0 * dW1 * np.sqrt(dt))
                
                # Antithetic variance reduction technique
                Vt[1] += kappa * (theta - Vt[1]) * dt + sigma * sqrt_Vt1 * anti_dW2 * np.sqrt(dt) + 0.25 * sigma**2 * dt * (anti_dW2**2 - 1)
                St[1] *= np.exp((r - 0.5 * Vt[1]) * dt + sqrt_Vt1 * anti_dW1 * np.sqrt(dt))
                
                path[0][step] = St[0]
                path[1][step] = St[1]
            
            # Calculate payoff for original and antithetic paths
            sum_payoffs += (self.option.payoff(path[0]) + self.option.payoff(path[1])) / 2
        
        # Discount payoffs to present value
        price = np.exp(-r * T) * (sum_payoffs / self.nb_paths)
        
        return price
