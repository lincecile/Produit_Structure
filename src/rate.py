#%% imports

import scipy
import math
from dataclasses import dataclass
from typing import Optional, Dict

from time_utils.maturity import Maturity

#%% classes

@dataclass
class RateType:
    compounded = 'compounded'
    continuous = 'continuous'

@dataclass
class InterpolationType:
    linear = 'linear'
    cubic = 'cubic'
    quadractic = 'quadratic'

class Rate:
    def __init__(
        self,
        rate: Optional[float] = None,
        rate_type: RateType = RateType.continuous,
        rate_curve: Optional[Dict[Maturity, float]] = None,
        interpolation_type: InterpolationType = InterpolationType.linear
    ) -> None:
        self.__rate = rate
        self.__rate_type = rate_type
        if rate_curve is not None:
            self.__interpol = scipy.interpolate.interp1d(
                [mat.maturity_in_years for mat in rate_curve.keys()],
                list(rate_curve.values()),
                fill_value="extrapolate",
                kind=interpolation_type,
            )

    def get_rate(self, maturity: Optional[Maturity] = None) -> float:
        if self.__rate is not None:
            return self.__rate
        if maturity is not None:
            return float(self.__interpol(maturity.maturity_years))
        raise ValueError('Please provide a valid maturity or a rate attribute.')

    def discount_factor(
        self, maturity: Maturity, force_rate: Optional[float] = None
    ) -> float:

        if self.__rate_type == RateType.continuous:
            return math.exp(
                -(
                    self.get_rate(maturity=maturity)
                    if force_rate is None
                    else force_rate
                )
                * maturity.maturity_years
            )
        elif self.__rate_type == RateType.compounded:
            return 1.0 / (
                (
                    1
                    + (
                        self.get_rate(maturity=maturity)
                        if force_rate is None
                        else force_rate
                    )
                )
                ** maturity.maturity_years
            )
            
    def __add__(self, other: float) -> float:
        return self.__rate + other if self.__rate is not None else other