#%% imports

from dataclasses import dataclass

#%% classes

@dataclass
class DayCount():
    act_365 = 365
    act_360 = 360