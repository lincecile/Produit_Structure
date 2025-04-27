# Time Utilities Documentation

## Overview

The Time Utilities module provides classes for handling dates, maturities, and payment schedules in financial applications. These tools are essential for pricing financial instruments, calculating cash flows, and managing time-based events.

## Class Hierarchy

```
DayCount
│
Maturity
└── Schedule
```

## DayCount Class (day_count.py)

The [`DayCount`](../../src/time_utils/day_count.py#L7) class provides constants for different day count conventions used in financial calculations:

```python
@dataclass
class DayCount():
    act_365 = 365
    act_360 = 360
```

These conventions define how to calculate year fractions when dealing with interest rates, discount factors, and other time-based financial calculations.

## Maturity Class (maturity.py)

The [`Maturity`](../../src/time_utils/maturity.py#L7) class represents the time period between two dates, typically used to define the lifetime of a financial instrument:

```python
class Maturity:
    def __init__(self, start_date: Optional[dt.date] = None,
                 end_date: Optional[dt.date] = None,
                 day_count: DayCount = DayCount.act_365) -> None:
        # Initialize with date range and day count convention
```

### Key Features:

- Stores start and end dates
- Calculates the maturity in years based on the day count convention
- Validates that dates are provided and logically ordered

### Core Methods:

- [`__get_maturity()`](../../src/time_utils/maturity.py#L24): Calculates the maturity in years. It uses the day count convention to determine the fraction of the year between the start and end dates.
- [`__assert_non_nulity()`](../../src/time_utils/maturity.py#L17): Validates date inputs, making sure they are not None and that the start date is before the end date.

## Schedule Class (schedule.py)

The [`Schedule`](../../src/time_utils/schedule.py#L10) class extends [`Maturity`](../../src/time_utils/maturity.py#L7) to represent a series of payment or observation dates:

```python
class Schedule(Maturity):
    def __init__(self, start_date: Optional[dt.date] = None,
                 end_date: Optional[dt.date] = None,
                 day_count: DayCount = DayCount.act_365,
                 frequency: int = 1) -> None:
        # Initialize with date range and payment frequency
```

### Key Features:

- Inherits maturity calculation from the [`Maturity`](../../src/time_utils/maturity.py#L7) class
- Generates a list of dates based on the specified frequency
- Calculates accrual periods between scheduled dates

### Core Methods:

- [`__get_schedule()`](../../src/time_utils/schedule.py#L19): Generates the list of dates in the schedule
- [`get_accrual()`](../../src/time_utils/schedule.py#L21): Calculates the accrual factor for a specific payment date

## Usage in Financial Products

The time utilities are essential components used by various financial product classes:

### In Bonds

```python
import datetime as dt

from time_utils.maturity import Maturity
from time_utils.day_count import DayCount
from rate import Rate
from bonds import Bond

# Create a 5-year bond with semi-annual coupons
today = dt.date.today()
maturity_date = today + dt.timedelta(days=365*5)
maturity = Maturity(start_date=today, end_date=maturity_date, day_count=DayCount.act_365)
rate = Rate(0.04)  # 4% interest rate

bond = Bond(
    name="5Y Corporate Bond",
    rate=rate,
    maturity=maturity,
    face_value=1000.0,
    coupon=0.04,
    nb_coupon=2  # Semi-annual coupon payments
)
```

### In Swaps

```python
import datetime as dt

from time_utils.schedule import Schedule
from time_utils.day_count import DayCount
from rate import Rate
from swaps import Swap, LegType

# Create a 3-year quarterly swap
today = dt.date.today()
end_date = today + dt.timedelta(days=365*3)

schedule = Schedule(
    start_date=today,
    end_date=end_date,
    day_count=DayCount.act_365,
    frequency=90  # Quarterly payments (approximately)
)

fixed_rate = Rate(0.035)  # 3.5% fixed rate
floating_rate = Rate(0.025)  # Initial floating rate

swap = Swap(
    name="3Y Fixed-for-Floating",
    notional=1_000_000,
    schedule=schedule,
    receiving_leg=LegType.FIXED,
    paying_leg=LegType.FLOATING,
    receiving_rate=fixed_rate,
    paying_rate=floating_rate
)
```


## Summary

The time utilities module provides essential tools for handling dates, calculating time periods, and managing payment schedules in financial applications. These classes integrate seamlessly with the financial product classes to enable accurate pricing and cash flow calculations.