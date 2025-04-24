# hestonpricer/models/option_base.py
from abc import ABC, abstractmethod

class OptionBase(ABC):
    def __init__(self, spot_price, strike, maturity, risk_free_rate, is_call):
        """
        Base class for options
        
        Args:
            spot_price: Initial price of the underlying asset
            strike: Strike price of the option
            maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate
            is_call: Boolean indicating if the option is a call option (True) or put option (False)
        """
        self.spot_price = spot_price
        self.strike = strike
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.is_call = is_call
    
    @abstractmethod
    def payoff(self, path):
        """
        Calculate the payoff of the option for a given price path
        
        Args:
            path: Array of prices simulating the path of the underlying asset
            
        Returns:
            float: The payoff of the option
        """
        pass
