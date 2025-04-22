#%% imports
from abc import abstractmethod
from dataclasses import dataclass
import datetime as dt
from typing import Optional

from products import Product
from timing.schedule import Schedule
from rate import Rate

#%% classes

@dataclass
class LegType:
    fixed = 'fixed'
    floating = 'floating'
    basket = 'basket'

class SwapBase(Product):
    """
    Base class to represent a swap.
    """
    def __init__(self,
                 name : str, 
                 notional: float,
                 schedule: Schedule,
                 receiving_leg : LegType,
                 paying_leg : LegType,
                 price_history: Optional[list[float]] = None
                 ) -> None:
        super().__init__(name, price_history)

        self.notional = notional
        self.schedule = schedule
        self.receiving_leg = receiving_leg
        self.paying_leg = paying_leg

        self.price = self._get_price()

    @abstractmethod
    def _get_price(self) -> float:
        """
        Calculate the price of the product.
        """
        pass
    
class Swap(SwapBase):
    """
    Class representing a vanilla rate swap.
    """
    def __init__(self,
                 name : str,
                 notional: float, 
                 schedule: Schedule,
                 receiving_leg : LegType,
                 paying_leg : LegType,
                 receiving_rate: Rate,
                 paying_rate: Rate,
                 receiving_rate_premium: float = 0.0,
                 paying_rate_premium: float = 0.0,
                 price_history: Optional[list[float]] = None
                 ) -> None:
        
        self.notional = notional
        self.receiving_rate = receiving_rate
        self.receiving_rate_premium = receiving_rate_premium
        self.total_receiving_rate = receiving_rate + receiving_rate_premium
        self.paying_rate = paying_rate
        self.paying_rate_premium = paying_rate_premium
        self.total_paying_rate = paying_rate + paying_rate_premium

        super().__init__(name=name,
                         price_history=price_history, 
                         notional=notional, 
                         schedule=schedule, 
                         receiving_leg=receiving_leg, 
                         paying_leg=paying_leg)


    def _get_price(self) -> float:
        """
        Calculate the price of the vanilla swap as the PV difference between the receiving 
        and paying legs. For each payment date, we accumulate the accrual factors to represent 
        the elapsed time from the start date, and use that to compute a discount factor.
  
        Returns:
            float: The swap price.
        """
        if self.price is not None:
            return self.price
        
        price = 0.0
        r = self.receiving_rate.get_rate()
        cumulative_accrual = 0.0
        
        for payment_date in self.schedule.schedule:
            period_accrual = self.schedule.get_accrual(payment_date)
            cumulative_accrual += period_accrual
            discount_factor = 1 / (1 + r * cumulative_accrual)
            receiving_cf = self.notional * self.total_receiving_rate * period_accrual * discount_factor
            paying_cf = self.notional * self.total_paying_rate * period_accrual * discount_factor
            price += (receiving_cf - paying_cf)
            
        return price