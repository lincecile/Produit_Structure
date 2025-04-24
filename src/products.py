# %% imports

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

# %%classes

class Product(ABC):
    """
    Abstract base class for financial products.
    Assumes that all products have a name, a daily price history, and a price.
    """
    def __init__(self, name: str, 
                 price_history: Optional[np.array] = None,
                 price : Optional[float] = None,
                 volatility : Optional[float] = None
                 ) -> None:
        self.name = name
        self.price_history = price_history if price_history is not None else np.array([])
        self.price = price
        self.volatility = volatility if volatility is not None else self.__get_volatility()
        
    @abstractmethod
    def _get_price(self) -> float:
        """
        Calculate the price of the product.
        """
        pass
    
    def __get_volatility(self) -> float:
        """
        Calculate the volatility of the product.
        """
        if self.price_history.size == 0:
            return 0.0
        return np.std(self.price_history) * np.sqrt(252) #TODO: check if 252 is the right number of days in a year for the product