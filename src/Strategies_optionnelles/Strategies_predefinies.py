import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional
import datetime as dt
from src.options.Pricing_option.Classes_Both.module_option import Option
from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche

# Import de la classe OptionsPortfolio depuis le fichier options_portfolio.py
from src.Strategies_optionnelles.Portfolio_options import OptionsPortfolio

class OptionsStrategy:
    """
    Classe permettant de créer des stratégies d'options prédéfinies
    et de les ajouter à un portefeuille existant.
    """
    
    def __init__(self, portfolio : OptionsPortfolio, market_data: DonneeMarche, expiry_date=None, underlying=None):
        """
        Initialise un gestionnaire de stratégies d'options.
        """
        self.portfolio = portfolio
        self.market_data = market_data
        self.expiry_date = expiry_date
        self.underlying = underlying or "default"
    
    def create_option(self, strike: float, is_call: bool, americaine: bool = False) -> Option:
        """
        Crée un objet Option avec les paramètres spécifiés.
        """
        return Option(
            prix_exercice=strike,
            maturite=self.expiry_date,
            call=is_call,
            americaine=americaine
        )

    def long_call(self, strike: float, quantity: float = 1.0, americaine: bool = False) -> None:
        """
        Ajoute un(des) call(s) long(s) au portefeuille.
        """
        option = self.create_option(strike, is_call=True, americaine=americaine)
        self.portfolio.add_option(option, quantity)
        
    def short_call(self, strike: float, quantity: float = 1.0, americaine: bool = False) -> None:
        """
        Ajoute un(des) call(s) short(s) au portefeuille.
        """
        option = self.create_option(strike, is_call=True, americaine=americaine)
        self.portfolio.add_option(option, -quantity)
    
    def long_put(self, strike: float, quantity: float = 1.0, americaine: bool = False) -> None:
        """
        Ajoute un(des) put(s) long(s) au portefeuille.
        """
        option = self.create_option(strike, is_call=False, americaine=americaine)
        self.portfolio.add_option(option, quantity)
    
    def short_put(self, strike: float, quantity: float = 1.0, americaine: bool = False) -> None:
        """
        Ajoute un(des) put(s) short(s) au portefeuille.
        """
        option = self.create_option(strike, is_call=False, americaine=americaine)
        self.portfolio.add_option(option, -quantity)

    def call_spread(self, lower_strike: float, upper_strike: float, quantity: float = 1.0, americaine: bool = False) -> None:
        """
        Crée un bull call spread: achat d'un call à strike bas, vente d'un call à strike haut.
        
        Args:
            lower_strike: Strike du call long (achat)
            upper_strike: Strike du call short (vente)
            quantity: Nombre de spreads
            americaine: True pour options européennes, False pour américaines
        """
        if lower_strike >= upper_strike:
            raise ValueError("Le strike inférieur doit être plus petit que le strike supérieur pour un call spread")
        
        self.long_call(lower_strike, quantity, americaine)
        self.short_call(upper_strike, quantity, americaine)
    
    def put_spread(self, lower_strike: float, upper_strike: float, quantity: float = 1.0, americaine: bool = False) -> None:
        """
        Crée un bear put spread: achat d'un put à strike haut, vente d'un put à strike bas.
        
        Args:
            lower_strike: Strike du put short (vente)
            upper_strike: Strike du put long (achat)
            quantity: Nombre de spreads
            americaine: True pour options européennes, False pour américaines
        """
        if lower_strike >= upper_strike:
            raise ValueError("Le strike inférieur doit être plus petit que le strike supérieur pour un put spread")
        
        self.short_put(lower_strike, quantity, americaine)
        self.long_put(upper_strike, quantity, americaine)
    
    def strangle(self, put_strike: float, call_strike: float, quantity: float = 1.0, americaine: bool = False) -> None:
        """
        Crée un strangle: achat d'un put OTM et d'un call OTM.
        
        Args:
            put_strike: Strike du put (doit être < prix actuel)
            call_strike: Strike du call (doit être > prix actuel)
            quantity: Nombre de strangles
            americaine: True pour options européennes, False pour américaines
        """
        current_price = self.market_data.prix_spot
        
        if not (put_strike < current_price < call_strike):
            raise ValueError(f"Pour un strangle, le strike du put ({put_strike}) doit être inférieur au prix actuel ({current_price}) et le strike du call ({call_strike}) supérieur")
        
        self.long_put(put_strike, quantity, americaine)
        self.long_call(call_strike, quantity, americaine)
    
    def straddle(self, strike: float, quantity: float = 1.0, americaine: bool = False) -> None:
        """
        Crée un straddle: achat d'un put et d'un call au même strike (généralement ATM).
        
        Args:
            strike: Strike pour les deux options
            quantity: Nombre de straddles
            americaine: True pour options européennes, False pour américaines
        """
        self.long_put(strike, quantity, americaine)
        self.long_call(strike, quantity, americaine)
    
    def butterfly(self, lower_strike: float, middle_strike: float, upper_strike: float, 
                  quantity: float = 1.0, is_call: bool = True, americaine: bool = False) -> None:
        """
        Crée un butterfly spread: achat d'une option au strike bas, 
        vente de deux options au strike milieu, achat d'une option au strike haut.
        
        Args:
            lower_strike: Strike le plus bas
            middle_strike: Strike du milieu
            upper_strike: Strike le plus haut
            quantity: Nombre de butterflies
            is_call: True pour utiliser des calls, False pour des puts
            americaine: True pour options européennes, False pour américaines
        """
        if not (lower_strike < middle_strike < upper_strike):
            raise ValueError("Les strikes doivent être en ordre croissant pour un butterfly")
            
        # Vérifier que les strikes sont équidistants
        if abs((middle_strike - lower_strike) - (upper_strike - middle_strike)) > 0.0001:
            print("ATTENTION: Les strikes ne sont pas équidistants, ce qui est généralement recommandé pour un butterfly")
        
        if is_call:
            self.long_call(lower_strike, quantity, americaine)
            self.short_call(middle_strike, 2 * quantity, americaine)
            self.long_call(upper_strike, quantity, americaine)
        else:
            self.long_put(lower_strike, quantity, americaine)
            self.short_put(middle_strike, 2 * quantity, americaine)
            self.long_put(upper_strike, quantity, americaine)
    
    def collar(self, put_strike: float, call_strike: float, 
               option_quantity: float = None, americaine: bool = False) -> None:
        """
        Crée un collar: protection d'une position sous-jacente longue 
        en achetant un put et en vendant un call.
        
        Args:
            stock_quantity: Quantité du sous-jacent détenue (doit être positive)
            put_strike: Strike du put (protection)
            call_strike: Strike du call (vendu pour financer le put)
            option_quantity: Quantité d'options (si différente de stock_quantity)
            americaine: True pour options européennes, False pour américaines
        """

        if put_strike >= call_strike:
            raise ValueError("Le strike du put doit être inférieur au strike du call pour un collar")

        self.long_put(put_strike, option_quantity, americaine)
        self.short_call(call_strike, option_quantity, americaine)
        strike_forward = (put_strike + call_strike) / 2
        self.forward(strike_forward, option_quantity, americaine)
    
    def forward(self, strike_forward: float,
               option_quantity: float = None, americaine: bool = False) -> None:
        """
        Crée un collar: protection d'une position sous-jacente longue 
        en achetant un put et en vendant un call.
        
        Args:
            stock_quantity: Quantité du sous-jacent détenue (doit être positive)
            put_strike: Strike du put (protection)
            call_strike: Strike du call (vendu pour financer le put)
            option_quantity: Quantité d'options (si différente de stock_quantity)
            americaine: True pour options européennes, False pour américaines
        """
        self.short_put(strike_forward, option_quantity, americaine)
        self.long_call(strike_forward, option_quantity, americaine)

    def create_strategy(self, strategy_name: str, params: dict, quantity_multiplier: float = 1.0) -> None:
        """
        Crée une stratégie d'options prédéfinie en fonction du nom de la stratégie
        et des paramètres fournis.
        
        Args:
            strategy_name: Nom de la stratégie à créer
            params: Dictionnaire contenant les paramètres nécessaires à la création de la stratégie
        
        Returns:
            None
        """
        strategy_name = strategy_name.lower()

        americaine = params.get("americaine", True)
        quantity = params.get("quantity", 1.0)

        quantity = quantity * quantity_multiplier
        
        # Appeler la méthode correspondante avec les bons paramètres
        if strategy_name == "call spread":
            self.call_spread(
                lower_strike=params["strike1"],
                upper_strike=params["strike2"],
                quantity=quantity,
                americaine=americaine
            )
        elif strategy_name == "put spread":
            self.put_spread(
                lower_strike=params["strike1"],
                upper_strike=params["strike2"],
                quantity=quantity,
                americaine=americaine
            )
        elif strategy_name == "strangle":
            self.strangle(
                put_strike=params["strike1"],
                call_strike=params["strike2"],
                quantity=quantity,
                americaine=americaine
            )
        elif strategy_name == "straddle":
            self.straddle(
                strike=params["strike"],
                quantity=quantity,
                americaine=americaine
            )
        elif strategy_name == "butterfly":
            self.butterfly(
                lower_strike=params["strike1"],
                middle_strike=params["strike2"],
                upper_strike=params["strike3"],
                quantity=quantity,
                is_call=params["is_call"],
                americaine=americaine
            )
        elif strategy_name == "collar":
            self.collar(
                put_strike=params["strike1"],
                call_strike=params["strike2"],
                option_quantity=quantity,
                americaine=americaine
            )
        elif strategy_name == "forward":
            self.forward(
                strike_forward=params["strike"],
                option_quantity=quantity,
                americaine=americaine
            )
        