#%% imports

import datetime as dt
from typing import Optional

from .day_count import DayCount

#%% classes

class Maturity:
    def __init__(self, start_date : Optional[dt.date] = None ,
                 end_date : Optional[dt.date] = None,
                 day_count : DayCount = DayCount.act_365) -> None :
        
        self.day_count = day_count
        self.start_date = start_date
        self.end_date = end_date
        
        self.__assert_non_nulity()
        
        self.maturity_years = self.__get_maturity()
        
    def __assert_non_nulity(self) -> None:
        test1 = (self.start_date is not None) & (self.end_date is not None)
        test2 = self.start_date <= self.end_date
        test = test1 * test2
        assert test, 'Please enter valid input options for start'
        
    def __get_maturity(self) -> float:
        days = (self.end_date - self.start_date).days
        # Convert day_count from string to numeric value if needed
        if isinstance(self.day_count, str):
            if self.day_count.upper() == "ACT/365":
                divisor = 365
            elif self.day_count.upper() == "ACT/360":
                divisor = 360
            else:
                raise ValueError(f"Unsupported day count convention: {self.day_count}")
        else:
            divisor = self.day_count
        return days / divisor
    
    def __repr__(self):
        return f'Maturity(start_date={self.start_date}, end_date={self.end_date}, maturity_years={self.maturity_years})'