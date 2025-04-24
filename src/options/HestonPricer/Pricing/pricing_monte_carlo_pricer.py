# hestonpricer/pricing/monte_carlo_pricer.py
import numpy as np
from Pricing.pricing_pricer_base import PricerBase
from Utils.utils_random_number_generator import RandomNumberGenerator
from Utils.utils_stats import Stats
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
            nb_steps: Number of time steps for discretization
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
        S0 = self.option.spot_price
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
    
    def _price_with_seed(self, seed):
        """Helper function for parallel processing"""
        return self.price(random_seed=seed)
    
    def price_multiple(self, nb_prices):
        """
        Calculate option price multiple times to estimate confidence interval
        
        Args:
            nb_prices: Number of times to calculate price
            
        Returns:
            list: [mean_price, margin_of_error, std_dev]
        """
        # Use parallel processing to speed up calculations
        with Pool() as pool:
            prices = pool.map(self._price_with_seed, range(nb_prices))
        
        prices = np.array(prices)
        mean_price = np.mean(prices)
        std_dev = np.std(prices, ddof=1)  # Sample standard deviation
        margin_of_error = 1.96 * std_dev / np.sqrt(nb_prices)
        
        return [mean_price, margin_of_error, std_dev]
    
    def price_over_parameter(self, parameter, min_val, max_val, steps, nb_prices):
        """
        Calculate option prices for different parameter values with confidence intervals
        
        Args:
            parameter: Parameter name to vary
            min_val: Minimum parameter value
            max_val: Maximum parameter value
            steps: Number of steps between min and max
            nb_prices: Number of times to calculate price for each parameter value
            
        Returns:
            numpy.ndarray: Array of parameter values, prices, and margins of error
        """
        result = np.zeros((steps + 1, 3))
        original_value = self._get_variable_value(parameter)
        
        for i in range(steps + 1):
            param_value = min_val + (max_val - min_val) * (i / steps)
            self._set_variable_value(parameter, param_value)
            result[i, 0] = self._get_variable_value(parameter)
            
            price_infos = self.price_multiple(nb_prices)
            result[i, 1] = price_infos[0]  # Mean price
            result[i, 2] = price_infos[1]  # Margin of error
        
        # Restore original value
        self._set_variable_value(parameter, original_value)
        return result
    
    def sensi_over_parameter(self, sensi, parameter, min_val, max_val, steps, nb_draws):
        """
        Calculate sensitivity over parameter range with confidence intervals
        
        Args:
            sensi: Sensitivity parameter to calculate
            parameter: Parameter name to vary
            min_val: Minimum parameter value
            max_val: Maximum parameter value
            steps: Number of steps between min and max
            nb_draws: Number of draws for sensitivity calculation
            
        Returns:
            numpy.ndarray: Array of parameter values, sensitivities, and margins of error
        """
        result = np.zeros((steps + 1, 3))
        original_value = self._get_variable_value(parameter)
        
        for i in range(steps + 1):
            param_value = min_val + (max_val - min_val) * (i / steps)
            self._set_variable_value(parameter, param_value)
            result[i, 0] = self._get_variable_value(parameter)
            
            fod = self.first_order_derivative_multiple(sensi, 1, nb_draws)
            result[i, 1] = fod[0]  # Mean sensitivity
            result[i, 2] = fod[1]  # Margin of error
        
        # Restore original value
        self._set_variable_value(parameter, original_value)
        return result
