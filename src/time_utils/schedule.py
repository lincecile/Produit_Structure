#%% imports

import datetime as dt
from typing import Optional

from .day_count import DayCount
from .maturity import Maturity

#%% classes

class Schedule(Maturity):
    def __init__(self, start_date: Optional[dt.date] = None,
                 end_date: Optional[dt.date] = None,
                 day_count: DayCount = DayCount.act_365,
                 frequency: int = 1) -> None:
        super().__init__(start_date, end_date, day_count)
        self.frequency = frequency

        self.schedule = self.__get_schedule()

    def __get_schedule(self) -> list[dt.date]:
        """
        Generate a list of dates between start_date and end_date with the specified frequency.
        """

        schedule = []
        current_date = self.start_date
        while current_date < self.end_date:
            schedule.append(current_date)
            current_date += dt.timedelta(self.frequency)

        # Ensure the end_date is included in the schedule
        if schedule[-1] != self.end_date:
            schedule.append(self.end_date)

        return schedule
    
    def get_accrual(self, payment_date: dt.date) -> float:
        """
        Calculate the accrual factor for the period ending at payment_date.
        Assumes a simple year fraction using a 365-day year.
        """
        try:
            index = self.schedule.index(payment_date)
        except ValueError:
            raise ValueError("The payment_date is not in the schedule list.")
        if index == 0:
            return 0.0
        previous_date = self.schedule[index - 1]
        return (payment_date - previous_date).days / self.day_count

    def __repr__(self):
        return f'Schedule(start_date={self.start_date}, end_date={self.end_date}, frequency={self.frequency})'