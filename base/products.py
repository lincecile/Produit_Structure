#%% imports

from abc import ABC, abstractmethod

from maturity.maturity import Maturity
from rate import Rate

#%% global variables

#%%classes

class Product(ABC):
    """
    Abstract base class for financial products.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def price(self) -> float:
        """
        Calculate the price of the product.
        """
        pass
    
class BondBase(Product):
    def __init__(self, name : str, rate):
        super().__init__(name)
        
        self.rate = rate
        
    @abstractmethod
    def price(self) -> float:
        """
        Calculate the price of the product.
        """
        pass
    
    @abstractmethod
    def ytm