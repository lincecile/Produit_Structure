#%% imports
from abc import abstractmethod
from scipy.optimize import newton
from typing import Optional

import datetime as dt

from timing.maturity import Maturity
from rate import Rate
from products import Product

#%% classes

class BondBase(Product):
    """
    Base class to represent a bond.
    """
    def __init__(
        self, name: str, 
        rate: Rate,
        maturity: Maturity, 
        face_value: float,
        price_history: Optional[list[float]] = None
    ) -> None:
        super().__init__(name, price_history)  
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
        price_history: Optional[list[float]] = None
    ) -> None:
        super().__init__(
            name=name,
            rate=rate,
            maturity=maturity,
            face_value=face_value,
            price_history=price_history
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
        price_up = self.face_value * self.rate.discount_factor(self.maturity, force_rate=original_rate + delta)
        price_down = self.face_value * self.rate.discount_factor(self.maturity, force_rate=original_rate - delta)
        return (price_down - price_up) / 2

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
        price_history: Optional[list[float]] = None
    ) -> None:
        super().__init__(
            name=name,
            rate=rate,
            maturity=maturity,
            face_value=face_value,
            price_history=price_history
        )
        self.coupon = coupon
        self.nb_coupon = nb_coupon

    def _get_price(self) -> float:
        """
        Calculate the price of the bond.
        """
        if self.price is not None:
            return self.price
        return self.face_value * self.rate.discount_factor(self.maturity)

    def get_dv01(self) -> float:
        """
        Calculate the DV01 of the bond using a finite difference.
        """
        delta = 1e-4  # one basis point
        original_rate = self.rate.get_rate(self.maturity)
        price_up = self.face_value * self.rate.discount_factor(self.maturity, force_rate=original_rate + delta)
        price_down = self.face_value * self.rate.discount_factor(self.maturity, force_rate=original_rate - delta)
        return (price_down - price_up) / 2

    def get_ytm(self) -> float:
        """
        Calculate the yield to maturity (YTM) of the bond.
        """
        def f(ytm):
            return (
                (self.coupon * self.face_value / self.nb_coupon)
                * (
                    1
                    - (1 + ytm / self.nb_coupon)
                    ** (-self.nb_coupon * self.maturity.maturity_years)
                )
                + self.face_value
                * (1 + ytm / self.nb_coupon)
                ** (-self.nb_coupon * self.maturity.maturity_years)
                - self.price
            )
        return newton(f, 0.05)
    
    def strip_bond(self) -> list:
        """
        Perform bond stripping to calculate the cash flows of the bond.
        Returns a list of tuples where each tuple contains (time, cash_flow).
        """
        cash_flows = []
        for i in range(1, int(self.nb_coupon * self.maturity.maturity_years) + 1):
            time = i / self.nb_coupon
            if i == self.nb_coupon * self.maturity.maturity_years:
                cash_flow = self.coupon * self.face_value / self.nb_coupon + self.face_value
            else:
                cash_flow = self.coupon * self.face_value / self.nb_coupon
            cash_flows.append((time, cash_flow))
        return cash_flows
    
    def to_zcbonds(self) -> list[ZCBond]:
        """
        Strip the bond into its component zero-coupon bonds.
        Each ZCBond corresponds to a single cash flow occurring at a future date.
        """
        zc_bonds = []
        cash_flows = self.strip_bond()
        
        for i, (time, cash) in enumerate(cash_flows):
            delta_days = round(time * self.maturity.day_count)
            start_date = self.maturity.start_date
            end_date = start_date + dt.timedelta(days=delta_days)

            maturity = Maturity(
                start_date=start_date,
                end_date=end_date,
                day_count=self.maturity.day_count,
            )

            zc_bond = ZCBond(
                name=f"{self.name}_ZC_{i+1}",
                rate=self.rate,
                maturity=maturity,
                face_value=cash,
            )

            zc_bonds.append(zc_bond)
        
        return zc_bonds