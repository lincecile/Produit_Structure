# hestonpricer/pricing/pricer_base.py
from abc import ABC, abstractmethod
import numpy as np
# from src.options.HestonPricer.Models.models_heston_parameters import HestonParameters
from ..Models.models_heston_parameters import HestonParameters

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
    
   