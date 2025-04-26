#%% imports

import scipy
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
from typing import Union, Optional, Tuple, Dict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')

from time_utils.maturity import Maturity
from time_utils.day_count import DayCount

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
        self._rate_curve = rate_curve
        if rate_curve is not None:
            self.__interpol = scipy.interpolate.interp1d(
                [mat.maturity_years for mat in self._rate_curve.keys()],
                list(self._rate_curve.values()),
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
    
    def _get_stochastic_rate(self, num_paths: int = 1, 
                         a: float = None, b: float = None, sigma: float = None,
                         num_steps: int = 252, auto_calibrate: bool = False) -> 'Rate':  # Changed default to False
        """
        Simulates a stochastic interest rate path using the Vasicek model.
        Parameters:
            num_paths (int): The number of simulation paths to generate (default is 1).
            a (float, optional): The speed of mean reversion parameter. If not provided, defaults based on auto-calibration or other internal settings.
            b (float, optional): The long-term mean parameter. If not provided, defaults based on auto-calibration or other internal settings.
            sigma (float, optional): The volatility of the rate process. If not provided, defaults based on auto-calibration or other internal settings.
            num_steps (int): Number of simulation steps for the interest rate path (default is 252).
            auto_calibrate (bool): Flag indicating whether to automatically calibrate model parameters using the current rate object if available (default is False).
        Returns:
            Rate: A simulated rate path if num_paths is 1; otherwise, a collection of simulated rate paths.
        Notes:
            - The simulation uses the longest maturity available from the internal rate curve.
            - If auto_calibrate is True, the Vasicek model uses the current rate object's interpolation data (if available) to calibrate parameters.
        """
        
        rate_curve = self._rate_curve
        sorted_maturities = sorted(rate_curve.keys(), key=lambda m: m.maturity_years)
        maturity = sorted_maturities[-1]  # Use the longest maturity for simulation

        vasicek = Vasicek(
            a=a,
            b=b,
            sigma=sigma,
            t=maturity.maturity_years,
            num_steps=num_steps,
            num_paths=num_paths,
            auto_calibrate=auto_calibrate,
            rate_object=self if hasattr(self, '_Rate__interpol') else None
        )

        paths = vasicek.simulate()
        return paths[0] if num_paths == 1 else paths
    
    def __add__(self, other: float) -> float:
        return self.__rate + other if self.__rate is not None else other

class StochasticRate(Rate):
    
    def __init__(self, 
                 rate_curve: Dict[Maturity, float], 
                 rate: Optional[float] = None, 
                 rate_type: RateType = RateType.continuous,
                 interpolation_type: InterpolationType = InterpolationType.linear,
                 num_paths : int = 1, 
                 a: float = 0.2,  # Default mean reversion speed
                 b: float = None,
                 sigma: float = 0.02,  # Default volatility
                 num_steps: int = 252, 
                 auto_calibrate: bool = False) -> None: 
        super().__init__(rate = rate, rate_type = rate_type, rate_curve = rate_curve, interpolation_type = interpolation_type)
        
        self.num_paths = num_paths
        self.a = a
        if b is None: #using the last maturity rate as b
            b = rate_curve[[*rate_curve][-1]]
        self.b = b
        self.sigma = sigma
        self.num_steps = num_steps
        self.auto_calibrate = auto_calibrate
        
        self._rate_curve = self.__get_new_curve()

    def __get_new_curve(self) -> Dict[Maturity, float]:
        """
        Generate a new rate curve using the stochastic rate model.
        The curve is created by averaging across all simulated paths.
        
        Returns:
            Dict[Maturity, float]: A dictionary of maturities and their corresponding rates.
        """
        
        simulated_rates = self._get_stochastic_rate(num_paths=self.num_paths,
            a=self.a, b=self.b, sigma=self.sigma,
            num_steps=self.num_steps, auto_calibrate=self.auto_calibrate
        )
    
        all_dates = simulated_rates[0]._rate_curve.keys()
        mean_results = {}
        
        for mat in all_dates:
            mat_rate = []
            for path_rate in simulated_rates:
                # Get the rate for this maturity from each path
                rate = path_rate.get_rate(mat)
                mat_rate.append(rate)
            mat_rate = np.mean(mat_rate)
            mean_results[mat] = mat_rate
    
        return mean_results
    
    def get_curve(self) -> np.ndarray:
        """
        Get the rate curve as a numpy array.
        
        Returns:
            np.ndarray: The rate curve as a numpy array.
        """
        return np.array(list(self._rate_curve.values()))

class InterestRateModels(ABC):
    """
    Abstract base class for interest rate models.
    """
    @abstractmethod
    def simulate(self):
        """
        Simulate the interest rate paths.
        """
        pass

    @abstractmethod
    def calibrate(self):
        """
        Calibrate the model parameters.
        """
        pass
    
class Vasicek(InterestRateModels):
    """
    Vasicek interest rate model for simulating mean-reverting short rates.

    Attributes:
        r0 (float): Initial short rate.
        a (float): Speed of mean reversion.
        b (float): Long-term mean level.
        sigma (float): Volatility of the short rate.
        t (float): Total time horizon (in years).
        num_steps (int): Number of discrete time steps.
        num_paths (int): Number of simulation paths.
        rate_object (Rate, optional): Rate object with curve data.
    """
    def __init__(self,
                 r0: Optional[Union[int, float]] = None,
                 a: Optional[Union[int, float]] = None,
                 b: Optional[Union[int, float]] = None,
                 sigma: Optional[Union[int, float]] = None,
                 t: Union[int, float] = 1.0,
                 num_steps: int = 252,
                 num_paths: int = 1,
                 historical_rates: Optional[Union[np.ndarray, pd.Series]] = None,
                 auto_calibrate: bool = False,
                 rate_object: Optional[Rate] = None):
        """
        Initialize the model, with optional Rate object or historical data for calibration.

        Args:
            r0 (float, optional): Initial short rate.
            a (float, optional): Speed of mean reversion.
            b (float, optional): Long-term mean.
            sigma (float, optional): Volatility.
            t (float): Total time horizon (in years).
            num_steps (int): Number of time steps.
            num_paths (int): Number of simulation paths.
            historical_rates (array-like, optional): Time series of rates for calibration.
            auto_calibrate (bool): Whether to calibrate automatically using historical_rates.
            rate_object (Rate, optional): A Rate object that may contain a rate curve.
        """
        
        #TODO let sigma be a dependant of historical rates
        
        self.t = float(t)
        self.num_steps = num_steps
        self.delta_t = self.t / self.num_steps
        self.num_paths = num_paths
        self.rate_object = rate_object

        # Get initial rate from Rate object if provided
        if rate_object is not None:
            # If r0 is not explicitly provided, try to get it from rate_object
            if r0 is None:
                # Try to get the shortest maturity rate or the base rate
                try:                    
                    today = dt.date.today()
                    shortest_maturity = Maturity(today, today + dt.timedelta(days=1), DayCount.act_365)
                    r0 = rate_object.get_rate(shortest_maturity)
                except:
                    # If no maturity works, try to get the base rate
                    try:
                        r0 = rate_object.get_rate()
                    except:
                        raise ValueError("Could not determine r0 from rate_object.")
        
        # Store r0 regardless of auto_calibrate setting
        if not isinstance(r0, (int, float)):
            raise ValueError("r0 must be a number.")
        self.r0 = float(r0)

        # Handle historical rates from Rate curve if available
        if self.rate_object is not None and hasattr(self.rate_object, '_Rate__interpol'):
            if auto_calibrate:
                # Prevent calibration with yield curve data - it's conceptually wrong
                raise ValueError(
                    "Auto-calibration cannot be performed using yield curve data. "
                    "Provide historical time series data via 'historical_rates' parameter instead, "
                    "or set auto_calibrate=False."
                )
            self.historical_rates = None  # Don't use rate curve for calibration
        elif historical_rates is not None:
            if isinstance(historical_rates, pd.Series):
                self.historical_rates = historical_rates.dropna().values
            else:
                self.historical_rates = np.asarray(historical_rates)
            if self.historical_rates.ndim != 1:
                raise ValueError("historical_rates must be a 1D array or Series.")
        else:
            self.historical_rates = None

        # Auto calibration block
        if auto_calibrate:
            if self.historical_rates is None:
                raise ValueError("auto_calibrate is True but no historical rates data were provided.")
            self.a, self.b, self.sigma = self.calibrate()
        else:
            # Validate and assign manually given parameters
            for name, val in zip(["a", "b", "sigma"], [a, b, sigma]):
                if not isinstance(val, (int, float)):
                    raise ValueError(f"{name} must be a number.")
            if a <= 0 or sigma <= 0 or t <= 0:
                raise ValueError("a, sigma, and t must be strictly positive.")

            self.a = float(a)
            self.b = float(b)
            self.sigma = float(sigma)

    def simulate(self) -> 'Rate':
        """
        Simulate short rate paths using the Vasicek model.

        Returns:
            Rate: Rate object containing the simulated rate paths.
        """
        from rate import Rate, RateType
        from time_utils.maturity import Maturity
        import datetime as dt
        from time_utils.day_count import DayCount

        # Simulate rates using the original method
        r = np.zeros((self.num_steps + 1, self.num_paths))
        r[0] = self.r0

        for i in range(1, self.num_steps + 1):
            epsilon = np.random.normal(loc=0, scale=1, size=self.num_paths)
            r[i] = r[i-1] + self.a * (self.b - r[i-1]) * self.delta_t + self.sigma * np.sqrt(self.delta_t) * epsilon

        # Create a Rate object with the simulated paths
        today = dt.date.today()
        
        # Create a dictionary of maturity points for each simulation step
        rate_curves = []
        
        for path_idx in range(self.num_paths):
            rate_curve = {}
            for step_idx in range(self.num_steps + 1):
                # Calculate the time point for this step
                step_time = step_idx * self.delta_t
                # Ensure at least one day difference for step_idx=0
                if step_idx == 0:
                    maturity_date = today + dt.timedelta(days=1)
                else:
                    maturity_date = today + dt.timedelta(days=max(1, int(365 * step_time)))
                maturity = Maturity(today, maturity_date, DayCount.act_365)
                
                # Store the rate for this maturity
                rate_curve[maturity] = r[step_idx, path_idx]
            
            # Create a Rate object for this path
            rate_obj = Rate(rate_type=RateType.continuous, rate_curve=rate_curve)
            rate_curves.append(rate_obj)
            
        # If we only have one path, return just that Rate object
        if self.num_paths == 1:
            return rate_curves[0]
        
        # Otherwise return a list of Rate objects (one per path)
        return rate_curves

    def calibrate(self) -> Tuple[float, float, float]:
        """
        Calibrate a, b, and sigma from the historical data using OLS (discrete Vasicek model).
        If a Rate object with rate_curve was provided, use that data for calibration.

        Returns:
            Tuple[float, float, float]: Calibrated values (a, b, sigma).
        """
        if self.historical_rates is None:
            raise ValueError("No historical data or rate curve provided for calibration.")

        # estimation
        x = self.historical_rates[:-1].reshape(-1, 1)
        y = self.historical_rates[1:]
        model = LinearRegression().fit(x, y)

        beta = model.coef_[0]
        alpha = model.intercept_

        a = (1 - beta) / self.delta_t
        b = alpha / (1 - beta)
        residuals = y - model.predict(x)
        sigma = np.std(residuals, ddof=1) / np.sqrt(self.delta_t)

        print("Model calibrated:")
        print(f"a={a:.6f} (mean reversion speed)")
        print(f"b={b:.6f} (long-term mean)")
        print(f"sigma={sigma:.6f} (volatility)")
        return a, b, sigma
