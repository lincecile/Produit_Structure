# hestonpricer/pricing/black_scholes_pricer.py
import numpy as np
import math
from ..Pricing.pricing_pricer_base import PricerBase
from ..Utils.utils_stats import Stats

class BlackScholesPricer(PricerBase):
    def __init__(self, option, volatility):
        """
        Black-Scholes pricer
        
        Args:
            option: Option object
            volatility: Volatility for Black-Scholes model
        """
        super().__init__(option, volatility=volatility)
    
    def price(self):
        """
        Calculate option price using Black-Scholes formula
        
        Returns:
            float: Option price
        """
        volatility = np.sqrt(self.heston_parameters.v0)
        S = self.option.spot_price
        K = self.option.strike
        r = self.option.risk_free_rate
        T = self.option.maturity
        
        d1 = (np.log(S / K) + (r + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        # Call price using Black-Scholes formula
        call_price = S * Stats.normal_cdf(d1) - K * np.exp(-r * T) * Stats.normal_cdf(d2)
        
        # Put price using put-call parity
        put_price = call_price - S + K * np.exp(-r * T)
        
        return call_price if self.option.is_call else put_price
