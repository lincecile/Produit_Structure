# hestonpricer/pricing/pricer_base.py
from abc import ABC, abstractmethod
import numpy as np
from Models.models_heston_parameters import HestonParameters

class PricerBase(ABC):
    def __init__(self, option, heston_parameters=None, volatility=None):
        """
        Base class for option pricing
        
        Args:
            option: Option object
            heston_parameters: Heston model parameters (optional)
            volatility: Volatility for Black-Scholes model (optional)
        """
        self.option = option
        
        if heston_parameters is not None:
            self.heston_parameters = heston_parameters
        elif volatility is not None:
            self.heston_parameters = HestonParameters(0, 0, volatility**2, 0, 0)
        else:
            raise ValueError("Either heston_parameters or volatility must be provided")
    
    @abstractmethod
    def price(self):
        """Calculate the price of the option"""
        pass
    
    def price_over_parameter(self, parameter, min_val, max_val, steps):
        """
        Calculate option prices for different parameter values
        
        Args:
            parameter: Parameter name to vary
            min_val: Minimum parameter value
            max_val: Maximum parameter value
            steps: Number of steps between min and max
            
        Returns:
            numpy.ndarray: Array of parameter values and option prices
        """
        result = np.zeros((steps + 1, 2))
        original_value = self._get_variable_value(parameter)
        
        for i in range(steps + 1):
            param_value = min_val + (max_val - min_val) * (i / steps)
            self._set_variable_value(parameter, param_value)
            result[i, 0] = self._get_variable_value(parameter)
            result[i, 1] = self.price()
        
        # Restore original value
        self._set_variable_value(parameter, original_value)
        return result
    
    def sensi_over_parameter(self, sensi, parameter, min_val, max_val, steps):
        """
        Calculate sensitivity over parameter range
        
        Args:
            sensi: Sensitivity parameter to calculate
            parameter: Parameter name to vary
            min_val: Minimum parameter value
            max_val: Maximum parameter value
            steps: Number of steps between min and max
            
        Returns:
            numpy.ndarray: Array of parameter values and sensitivities
        """
        result = np.zeros((steps + 1, 2))
        original_value = self._get_variable_value(parameter)
        
        for i in range(steps + 1):
            param_value = min_val + (max_val - min_val) * (i / steps)
            self._set_variable_value(parameter, param_value)
            result[i, 0] = self._get_variable_value(parameter)
            result[i, 1] = self.first_order_derivative(sensi, 0.01)
        
        # Restore original value
        self._set_variable_value(parameter, original_value)
        return result
    
    def _get_variable_value(self, variable):
        """Get a variable value from option or model parameters"""
        if variable == "spot_price":
            return self.option.spot_price
        elif variable == "strike":
            return self.option.strike
        elif variable == "maturity":
            return self.option.maturity
        elif variable == "risk_free_rate":
            return self.option.risk_free_rate
        elif variable == "kappa":
            if self.heston_parameters is None:
                raise ValueError("Heston parameters not set")
            return self.heston_parameters.kappa
        elif variable == "theta":
            if self.heston_parameters is None:
                raise ValueError("Heston parameters not set")
            return self.heston_parameters.theta
        elif variable == "sigma":
            if self.heston_parameters is None:
                raise ValueError("Heston parameters not set")
            return self.heston_parameters.sigma
        elif variable == "v0":
            if self.heston_parameters is None:
                raise ValueError("Heston parameters not set")
            return self.heston_parameters.v0
        elif variable == "rho":
            if self.heston_parameters is None:
                raise ValueError("Heston parameters not set")
            return self.heston_parameters.rho
        else:
            raise ValueError(f"Invalid variable name: {variable}")
    
    def _set_variable_value(self, variable, value):
        """Set a variable value in option or model parameters"""
        if variable == "spot_price":
            self.option.spot_price = value
        elif variable == "strike":
            self.option.strike = value
        elif variable == "maturity":
            self.option.maturity = value
        elif variable == "risk_free_rate":
            self.option.risk_free_rate = value
        elif variable == "kappa":
            if self.heston_parameters is None:
                raise ValueError("Heston parameters not set")
            self.heston_parameters.kappa = value
        elif variable == "theta":
            if self.heston_parameters is None:
                raise ValueError("Heston parameters not set")
            self.heston_parameters.theta = value
        elif variable == "sigma":
            if self.heston_parameters is None:
                raise ValueError("Heston parameters not set")
            self.heston_parameters.sigma = value
        elif variable == "v0":
            if self.heston_parameters is None:
                raise ValueError("Heston parameters not set")
            self.heston_parameters.v0 = value
        elif variable == "rho":
            if self.heston_parameters is None:
                raise ValueError("Heston parameters not set")
            self.heston_parameters.rho = value
        else:
            raise ValueError(f"Invalid variable name: {variable}")
    
    def first_order_derivative(self, variable, h=0.0001):
        """
        Calculate first order derivative (central difference)
        
        Args:
            variable: Variable name for derivative calculation
            h: Step size for finite difference
            
        Returns:
            float: First order derivative
        """
        original_value = self._get_variable_value(variable)
        
        # Forward step
        self._set_variable_value(variable, original_value + h)
        value_plus_h = self.price()
        
        # Backward step
        self._set_variable_value(variable, original_value - h)
        value_minus_h = self.price()
        
        # Restore original value
        self._set_variable_value(variable, original_value)
        
        # Central difference
        return (value_plus_h - value_minus_h) / (2 * h)
    
    def first_order_derivative_multiple(self, variable, h=0.0001, nb_draws=10):
        """
        Calculate first order derivative multiple times to estimate confidence interval
        
        Args:
            variable: Variable name for derivative calculation
            h: Step size for finite difference
            nb_draws: Number of times to calculate derivative
            
        Returns:
            list: [mean, margin_of_error, std_dev]
        """
        results = np.zeros(nb_draws)
        for i in range(nb_draws):
            results[i] = self.first_order_derivative(variable, h)
        
        mean = np.mean(results)
        std_dev = np.std(results, ddof=1)  # Sample standard deviation
        margin_of_error = 1.96 * std_dev / np.sqrt(nb_draws)
        
        return [mean, margin_of_error, std_dev]
    
    def second_order_derivative(self, variable, h=0.0001):
        """
        Calculate second order derivative
        
        Args:
            variable: Variable name for derivative calculation
            h: Step size for finite difference
            
        Returns:
            float: Second order derivative
        """
        original_value = self._get_variable_value(variable)
        
        # Forward step
        self._set_variable_value(variable, original_value + h)
        value_plus_h = self.price()
        
        # Backward step
        self._set_variable_value(variable, original_value - h)
        value_minus_h = self.price()
        
        # Original value
        self._set_variable_value(variable, original_value)
        value_original = self.price()
        
        # Second order central difference
        return (value_plus_h - 2 * value_original + value_minus_h) / (h * h)
