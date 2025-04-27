# %% imports
from abc import abstractmethod
import scipy.optimize as opt
from typing import Optional
import datetime as dt
import math

from src.time_utils.maturity import Maturity
from src.rate import Rate
from src.products import Product
from src.time_utils.day_count import DayCount


# %% classes


class BondBase(Product):
    """
    Base class to represent a bond.
    """

    def __init__(
        self,
        name: str,
        rate: Rate,
        maturity: Maturity,
        face_value: float,
        price_history: Optional[list[float]] = None,
        price: Optional[float] = None,
    ) -> None:
        super().__init__(name=name, price_history=price_history, price=price)
        self.rate = rate
        self.maturity = maturity
        self.face_value = face_value

        self.price = self._get_price()

    @abstractmethod
    def _get_price(self) -> float:
        """
        Calculate the price of the product.
        """
        pass

    @abstractmethod
    def get_duration(self) -> float:
        """
        Calculate the Macaulay duration of the bond.
        """
        pass

    @abstractmethod
    def get_dv01(self) -> float:
        """
        Calculate the DV01 of the product.
        """
        pass


class ZCBond(BondBase):
    """
    Class representing a zero-coupon bond.
    """

    def __init__(
        self,
        name: str,
        rate: Rate,
        maturity: Maturity,
        face_value: float,
        price_history: Optional[list[float]] = None,
        price: Optional[float] = None,
    ) -> None:
        super().__init__(
            name=name,
            rate=rate,
            maturity=maturity,
            face_value=face_value,
            price_history=price_history,
            price=price,
        )

    def _get_price(self) -> float:
        """
        Calculate the price of the zero-coupon bond.
        """
        if self.price is not None:
            return self.price
        return self.face_value * self.rate.discount_factor(self.maturity)

    def get_dv01(self) -> float:
        """
        Calculate the DV01 of the zero-coupon bond using a finite difference.
        """
        delta = 0.0001
        original_rate = self.rate.get_rate(self.maturity)
        price_up = self.face_value * self.rate.discount_factor(
            self.maturity, force_rate=original_rate + delta
        )
        price_down = self.face_value * self.rate.discount_factor(
            self.maturity, force_rate=original_rate - delta
        )
        return (price_down - price_up) / 2

    def get_duration(self) -> float:
        """
        Calculate the Macaulay duration of the zero-coupon bond.
        """
        # for a zero-coupon bond, the duration is equal to the maturity
        return self.maturity.maturity_years


class Bond(BondBase):
    """
    Class representing a bond.
    """

    def __init__(
        self,
        name: str,
        rate: Rate,
        maturity: Maturity,
        face_value: float,
        coupon: float,
        nb_coupon: int,
        price_history: Optional[list[float]] = None,
        price: Optional[float] = None,
    ) -> None:

        self.coupon = coupon
        self.nb_coupon = nb_coupon
        super().__init__(
            name=name,
            rate=rate,
            maturity=maturity,
            face_value=face_value,
            price_history=price_history,
            price=price,
        )

    def _get_price(self) -> float:
        """
        Calculate the price of the bond.
        """
        if self.price is not None:
            return self.price
        price = self.face_value * self.rate.discount_factor(self.maturity)
        coupon_value = self.__discounted_stripping(
            self.coupon,
            self.nb_coupon,
            self.maturity.maturity_years,
            self.face_value,
            self.maturity.start_date,
            self.maturity.day_count,
        )
        for _, cf in coupon_value:
            price += cf
        return price

    def __shock_price(self, delta: float) -> float:

        price = self.face_value * self.rate.discount_factor(
            self.maturity, force_rate=self.rate.get_rate(self.maturity) + delta
        )
        coupon_component = self.__discounted_stripping(
            self.coupon, self.nb_coupon, self.maturity.maturity_years, 
            self.face_value, self.maturity.start_date, self.maturity.day_count,
            force_rate=self.rate.get_rate(self.maturity) + delta)
        for _, cf in coupon_component:
            price += cf
        return price

    def get_dv01(self) -> float:
        """
        Calculate the DV01 of the bond using a finite difference.
        """
        delta = 1e-4  # one basis point
        price_up = self.__shock_price(delta)
        price_down = self.__shock_price(-delta)
        return (price_down - price_up) / 2

    def get_ytm(self) -> float:
        """
        Calculate the yield to maturity (YTM) of the bond.
        """
        def bond_price_error(ytm : float, price : float, face_value : float,
                             coupon : float, nb_coupon : int, 
                             maturity_years : float) -> float:
            periods = nb_coupon * maturity_years
            coupon_period = coupon * face_value / nb_coupon
            
            cf_sum = 0
            for i in range(1, int(periods) + 1):
                cf_sum += coupon_period / ((1 + ytm / nb_coupon) ** i)
            
            cf_sum += face_value / ((1 + ytm / nb_coupon) ** periods)
            return abs(cf_sum - price)
        
        result = opt.minimize(
            lambda ytm: bond_price_error(
                ytm, self.price, self.face_value, self.coupon, self.nb_coupon,
                self.maturity.maturity_years),
            x0=0.05,  # Initial guess for YTM
            method='SLSQP'
        )
        
        return round(result.x[0],4) if result.success else None

    def get_duration(self) -> float:
        """
        Calculate the Macaulay duration of the bond.
        """
        discounted_cash_flows = self.__discounted_stripping(
            self.coupon,
            self.nb_coupon,
            self.maturity.maturity_years,
            self.face_value,
            self.maturity.start_date,
            self.maturity.day_count,
        )
        discounted_face_value = self.face_value * self.rate.discount_factor(
            self.maturity
        )
        discounted_cash_flows[-1] = discounted_cash_flows[-1][0], discounted_cash_flows[-1][1] + discounted_face_value
        dcf_div_price_by_time = []
        for time, cash_flow in discounted_cash_flows:
            dcf_div_price_by_time.append(time * cash_flow / self.price)
        return sum(dcf_div_price_by_time)
        
    def __strip_bond(self,
        coupon: float, nb_coupon: int, maturity_years: float, face_value: float
    ) -> list:
        """
        Perform bond stripping to calculate the cash flows of the bond.
        Returns a list of tuples where each tuple contains (time, cash_flow).
        """
        cash_flows = []
        for i in range(1, int(nb_coupon * maturity_years) + 1):
            time = i / nb_coupon
            if i == nb_coupon * maturity_years:
                cash_flow = coupon * face_value / nb_coupon + face_value
            else:
                cash_flow = coupon * face_value / nb_coupon
            cash_flows.append((time, cash_flow))
        return cash_flows

    def __discounted_stripping(
        self,
        coupon: float,
        nb_coupon: int,
        maturity_years: float,
        face_value: float,
        start_date: dt.date,
        day_count: DayCount = DayCount.act_365,
        force_rate : Optional[float] = None,
    ) -> list:
        """
        Perform bond stripping to calculate the cash flows of the bond.
        Returns a list of tuples where each tuple contains (time, cash_flow).
        """
        strip = self.__strip_bond(coupon, nb_coupon, maturity_years, face_value)
        pv_cash_flows = []
        for time, cash_flow in strip:
            # Create a maturity object for this cash flow
            cf_maturity = Maturity(
                start_date=start_date,
                end_date=start_date + dt.timedelta(days=round(time * day_count)),
                day_count=day_count,
            )
            # Calculate the present value
            if force_rate is None:
                pv = cash_flow * self.rate.discount_factor(cf_maturity)
            else:
                pv = cash_flow * self.rate.discount_factor(cf_maturity, 
                                                           force_rate=force_rate)
            pv_cash_flows.append((time, pv))
        return pv_cash_flows

    def to_zcbonds(self) -> list[ZCBond]:
        """
        Strip the bond into its component zero-coupon bonds.
        Each ZCBond corresponds to a single cash flow occurring at a future date.
        """
        zc_bonds = []
        cash_flows = self.__strip_bond(
            self.coupon, self.nb_coupon, self.maturity.maturity_years, 
            self.face_value
        )
        
        for time, cash_flow in cash_flows:
            cf_maturity = Maturity(
                start_date=self.maturity.start_date,
                end_date=self.maturity.start_date
                + dt.timedelta(days=round(time * self.maturity.day_count)),
                day_count=self.maturity.day_count,
            )
            zc_bond = ZCBond(
                name=f"{self.name}_ZC_{time}",
                rate=self.rate,
                maturity=cf_maturity,
                face_value=cash_flow,
            )
            zc_bonds.append(zc_bond)
            
        return zc_bonds
