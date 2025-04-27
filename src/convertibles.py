#%% imports

from typing import Optional

from src.bonds import Bond
from src.time_utils.schedule import Schedule, Maturity
from src.rate import Rate
from src.stocks import Stock

from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche
from src.options.Pricing_option.Classes_Both.module_option import Option
from src.options.Pricing_option.Classes_Both.module_enums import ConventionBaseCalendaire, TypeBarriere, DirectionBarriere
from src.options.Pricing_option.Classes_TrinomialTree.module_arbre_noeud import Arbre
from src.options.Pricing_option.Classes_TrinomialTree.module_grecques_empiriques import GrecquesEmpiriques

#%% classes

class Convertible(Bond):
    """
    Class representing a convertible bond.
    """
    def __init__(self,
                 name: str,
                 rate: Rate,
                 maturity: Maturity,
                 face_value: float,
                 coupon: float,
                 nb_coupon: int,
                 conversion_ratio: float,
                 stock: Stock,
                 price_history: Optional[list[float]] = None,
                 price: Optional[float] = None) -> None:
        
        self.conversion_ratio = conversion_ratio
        self.stock = stock

        # Set temporary values so that when BondBase.__init__ calls _get_price, no error occurs.
        self.bond_component = 0
        self.option_component = 0

        super().__init__(name=name,
                         price_history=price_history, 
                         price=price, 
                         rate=rate, 
                         maturity=maturity,
                         face_value=face_value,
                         coupon=coupon,
                         nb_coupon=nb_coupon)
        
        # Reset the price so that _get_price() recomputes from updated components.
        self.price = None
        
        self.bond_component = self.__get_bond_component_price()
        self.option_component = self.__compute_option_price()
        
        # Recalculate price once all components are defined.
        self.price = self._get_price()
    
    def __get_bond_component_price(self) -> float:
        """
        Calculate the bond component of the convertible bond.
        
        Returns:
            float: The price of the bond component.
        """
        return super()._get_price()
    
    def _get_bond_component_duration(self) -> float:
        """
        Calculate the duration of the bond component of the convertible bond.
        
        Returns:
            float: The duration of the bond component.
        """
        return super().get_duration()
    
    def __initialize_option(self) -> Arbre:
        """
        Initialize the option component of the convertible bond.
        
        Returns:
            float: The initialized option component.
        """
        
        market_data = DonneeMarche(date_debut=self.maturity.start_date,
                                   prix_spot=self.stock.price,
                                   volatilite=self.stock.volatility,
                                   taux_interet=self.rate.get_rate(),
                                   taux_actualisation=self.rate.get_rate(),
                                   dividende_ex_date=self.stock.div_schedule.schedule[0],
                                   dividende_montant=self.stock.div_amount,
                                   dividende_rate=self.stock.div_amount / self.stock.price)
        
        strike_price = self.face_value / self.conversion_ratio
        option = Option(maturite=self.maturity.end_date,
                        prix_exercice=strike_price,
                        date_pricing=self.maturity.start_date)
        
        arbre = Arbre(nb_pas=500,
                      donnee_marche=market_data,
                      option=option)
        return arbre
    
    def __compute_option_price(self) -> float:
        """
        Compute the price of the option component of the convertible bond.
        
        Returns:
            float: The price of the option component.
        """
        arbre = self.__initialize_option()
        arbre.pricer_arbre()
        return arbre.prix_option
    
    def _get_option_component_duration(self) -> float:
        """
        Calculate the duration of the option component of the convertible bond.
        
        Returns:
            float: The duration of the option component.
        """
        greeks = GrecquesEmpiriques(arbre=self.__initialize_option())
        rho = greeks.approxime_rho() /100
        print(rho)
        price_change = self.option_component * rho
        return price_change
        
    def _get_price(self) -> float:
        """
        Calculate the price of the convertible bond.
        
        Returns:
            float: The price of the convertible bond, which is the sum of the bond and option components.
        """
        if self.price is not None:
            return self.price
        return self.bond_component + self.option_component
    
    def get_dv01(self) -> float:
        """
        Calculate the DV01 of the convertible bond.
        
        Returns:
            float: The DV01 of the convertible bond, which is the change in price for a 1 basis point change in yield.
        """
        actual_price = self._get_price()
        # Ensure the parameter order is correct: coupon and nb_coupon must be passed.
        new_convertible = Convertible(self.name,
                                      self.rate + 0.0001,
                                      self.maturity,
                                      self.face_value,
                                      self.coupon,
                                      self.nb_coupon,
                                      self.conversion_ratio,
                                      self.conversion_price,
                                      self.stock,
                                      self.price_history)
        new_price = new_convertible._get_price()
        return new_price - actual_price

    def get_duration(self) -> float:
        """
        Calculate the duration of the convertible bond by combining the weighted bond and option components.
        Returns:
            float: The duration of the convertible bond.
        """
        convertible_price = self.price
        bond_price = self.bond_component
        option_price = self.option_component
        bond_duration = self._get_bond_component_duration()
        option_duration = self._get_option_component_duration()
        
        weighted_bond_duration = bond_price / convertible_price * bond_duration
        weighted_option_duration = option_price / convertible_price * option_duration
        
        return weighted_bond_duration + weighted_option_duration