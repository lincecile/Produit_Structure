# hestonpricer/models/asian_option.py
import numpy as np
from src.options.HestonPricer.Models.models_option_base import OptionBase

class AsianOption(OptionBase):
    def __init__(self, spot_price, strike, maturity, risk_free_rate, is_call):
        """
        Asian option class
        
        Args:
            spot_price: Initial price of the underlying asset
            strike: Strike price of the option
            maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate
            is_call: Boolean indicating if the option is a call option (True) or put option (False)
        """
        super().__init__(spot_price, strike, maturity, risk_free_rate, is_call)
    
    def payoff(self, path):
        """
        Calculate the payoff of an Asian option based on average price
        
        Args:
            path: Array of prices simulating the path of the underlying asset
            
        Returns:
            float: The payoff of the Asian option
        """
        mean_price = np.mean(path)
        if self.is_call:
            return max(0, mean_price - self.strike)
        else:
            return max(0, self.strike - mean_price)
