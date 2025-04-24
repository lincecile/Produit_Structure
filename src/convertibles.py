#%% imports

from typing import Optional

from bonds import Bond
from time_utils.schedule import Schedule, Maturity
from rate import Rate
from stocks import Stock

from options.Pricing_option.Classes_Both.module_marche import DonneeMarche
from options.Pricing_option.Classes_Both.module_option import Option
from options.Pricing_option.Classes_Both.module_enums import ConventionBaseCalendaire, TypeBarriere, DirectionBarriere
from options.Pricing_option.Classes_TrinomialTree.module_arbre_noeud import Arbre

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
        
        self.bond_component = self.__get_bond_component()
        self.option_component = self.__compute_option()
        
        # Recalculate price once all components are defined.
        self.price = self._get_price()
    
    def __get_bond_component(self) -> float:
        """
        Calculate the bond component of the convertible bond.
        """
        return super()._get_price()
    
    def __compute_option(self) -> float:
        """
        Compute the option value of the convertible bond.
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
        arbre.pricer_arbre()
        return arbre.prix_option
    
    def _get_price(self) -> float:
        """
        Calculate the price of the convertible bond.
        """
        if self.price is not None:
            return self.price
        return self.bond_component + self.option_component
    
    def get_dv01(self):
        """
        Calculate the DV01 of the convertible bond.
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
