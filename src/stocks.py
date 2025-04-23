#%% imports

from typing import Optional

from products import Product
from time_utils.schedule import Schedule

#%% classes

class Stock(Product):
    """
    Class representing a stock.
    """
    def __init__(self, name: str,
                 price: float,
                 div_amount: float,
                 div_schedule: Schedule,
                 price_history: Optional[list[float]] = None,
                 volatility : Optional[float] = None
                 ) -> None:
        super().__init__(name=name,
                        price_history=price_history,
                        price=price,
                        volatility=volatility)
        self.div_amount = div_amount
        self.div_schedule = div_schedule
        
    def _get_price(self):
        return super()._get_price()