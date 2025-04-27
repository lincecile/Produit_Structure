import numpy as np
import datetime as dt
from typing import List, Dict, Tuple, Union, Optional

from src.options.Pricing_option.Classes_Both.module_option import Option
from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche
from src.options.Pricing_option.Classes_Both.module_barriere import Barriere, TypeBarriere, DirectionBarriere
from src.Strategies_optionnelles.Portfolio_options import OptionsPortfolio
from src.Strategies_optionnelles.Portfolio_structured import StructuredProductsPortfolio
from src.bonds import ZCBond, Bond
from src.rate import Rate
from src.time_utils.maturity import Maturity

class StructuredProductsStrategy:
    """
    Classe permettant de créer des produits structurés prédéfinis
    et de les ajouter à un portefeuille.
    """
    
    def __init__(self, 
                 structured_portfolio: StructuredProductsPortfolio, 
                 options_portfolio: OptionsPortfolio,
                 market_data: DonneeMarche, 
                 rate: Rate,
                 expiry_date=None, 
                 underlying=None):
        """
        Initialise un gestionnaire de produits structurés prédéfinis.
        
        Args:
            structured_portfolio: Le portefeuille de produits structurés
            options_portfolio: Le portefeuille d'options pour les composants
            market_data: Les données de marché pour le pricing
            rate: Le taux d'intérêt pour les obligations
            expiry_date: La date d'expiration du produit
            underlying: Le sous-jacent du produit
        """
        self.structured_portfolio = structured_portfolio
        self.options_portfolio = options_portfolio
        self.market_data = market_data
        self.rate = rate
        self.expiry_date = expiry_date
        self.underlying = underlying or "default"
        self.components = []  # Pour suivre les composants ajoutés
    
    def create_option(self, strike: float, is_call: bool, barrier_type=None, 
                     barrier_direction=None, barrier_level=None, americaine: bool = False) -> Option:
        """
        Crée un objet Option avec les paramètres spécifiés.
        
        Args:
            strike: Prix d'exercice de l'option
            is_call: True pour un call, False pour un put
            barrier_type: Type de barrière (knock_in, knock_out ou None)
            barrier_direction: Direction de la barrière (up, down ou None)
            barrier_level: Niveau de la barrière
            americaine: True pour une option américaine, False pour européenne
        
        Returns:
            Option: L'objet Option créé
        """
        barriere = Barriere(niveau_barriere=barrier_level,
                           type_barriere=barrier_type,
                           direction_barriere=barrier_direction) if barrier_type else None

        option = Option(
            prix_exercice=strike,
            barriere=barriere,
            maturite=self.expiry_date,
            call=is_call,
            americaine=americaine
        )
        
        return option
    
    def add_zero_coupon_bond(self, face_value: float, maturity: Maturity) -> ZCBond:
        """
        Ajoute une obligation zéro-coupon au produit structuré.
        
        Args:
            face_value: Valeur nominale de l'obligation
            maturity: Maturité de l'obligation
            
        Returns:
            ZCBond: L'obligation zéro-coupon créée
        """
        zc_bond = ZCBond(
            name=f"ZC_Bond_{len(self.components)}",
            rate=self.rate,
            maturity=maturity,
            face_value=face_value
        )
        
        self.components.append({
            'type': 'zero_coupon_bond',
            'object': zc_bond
        })
        
        return zc_bond
    
    def add_put_down_in(self, strike: float, barrier_level: float, quantity: float = 1.0) -> None:
        """
        Ajoute un put down-and-in au produit structuré.
        
        Args:
            strike: Prix d'exercice du put
            barrier_level: Niveau de la barrière
            quantity: Quantité d'options (positif pour position longue, négatif pour position courte)
        """
        put_option = self.create_option(
            strike=strike,
            is_call=False,
            barrier_type=TypeBarriere.knock_in,
            barrier_direction=DirectionBarriere.down,
            barrier_level=barrier_level
        )
        
        self.options_portfolio.add_option(put_option, quantity)
        
        self.components.append({
            'type': 'put_down_in',
            'object': put_option,
            'quantity': quantity
        })
    
    def add_digital_call(self, strike: float, payout: float, spread_width: float = 0.01, quantity: float = 1.0) -> None:
        """
        Ajoute une option digitale (call) au produit structuré, répliquée par un call spread.
        
        Args:
            strike: Prix d'exercice de l'option digitale
            payout: Montant du paiement si l'option est dans la monnaie
            spread_width: Écart entre les deux strikes du call spread
            quantity: Quantité d'options (positif pour position longue, négatif pour position courte)
        """
        # Calculer les strikes pour le call spread qui réplique l'option digitale
        lower_strike = strike
        upper_strike = strike + spread_width
        
        # Ajuster la quantité pour obtenir le payout souhaité
        adjusted_quantity = quantity * payout / spread_width
        
        # Créer un call spread pour répliquer l'option digitale
        call_lower = self.create_option(strike=lower_strike, is_call=True)
        call_upper = self.create_option(strike=upper_strike, is_call=True)
        
        # Ajouter les options au portefeuille
        self.options_portfolio.add_option(call_lower, adjusted_quantity)
        self.options_portfolio.add_option(call_upper, -adjusted_quantity)
        
        self.components.append({
            'type': 'digital_call',
            'strike': strike,
            'payout': payout,
            'spread_width': spread_width,
            'quantity': quantity
        })
    
    def add_digital_put(self, strike: float, payout: float, spread_width: float = 0.01, quantity: float = 1.0) -> None:
        """
        Ajoute une option digitale (put) au produit structuré, répliquée par un put spread.
        
        Args:
            strike: Prix d'exercice de l'option digitale
            payout: Montant du paiement si l'option est dans la monnaie
            spread_width: Écart entre les deux strikes du put spread
            quantity: Quantité d'options (positif pour position longue, négatif pour position courte)
        """
        # Calculer les strikes pour le put spread qui réplique l'option digitale
        upper_strike = strike
        lower_strike = strike - spread_width
        
        # Ajuster la quantité pour obtenir le payout souhaité
        adjusted_quantity = quantity * payout / spread_width
        
        # Créer un put spread pour répliquer l'option digitale
        put_upper = self.create_option(strike=upper_strike, is_call=False)
        put_lower = self.create_option(strike=lower_strike, is_call=False)
        
        # Ajouter les options au portefeuille
        self.options_portfolio.add_option(put_upper, adjusted_quantity)
        self.options_portfolio.add_option(put_lower, -adjusted_quantity)
        
        self.components.append({
            'type': 'digital_put',
            'strike': strike,
            'payout': payout,
            'spread_width': spread_width,
            'quantity': quantity
        })
    
    def capital_protected_note(self, 
                             protection_level: float = 1.0,
                             participation_rate: float = 1.0,
                             notional: float = 100.0,
                             cap: Optional[float] = None) -> None:
        """
        Crée une note à capital protégé (capital protected note).
        
        Args:
            protection_level: Niveau de protection du capital (1.0 = 100%)
            participation_rate: Taux de participation à la hausse du sous-jacent
            notional: Montant nominal de l'investissement
            cap: Plafond de rendement (None si pas de plafond)
        """
        current_price = self.market_data.prix_spot
        maturity_years = self.expiry_date.maturity_years
        
        # Réinitialiser les composants pour le nouveau produit
        self.components = []
        
        # 1. Ajouter l'obligation zéro-coupon pour la protection du capital
        maturity = Maturity(
            start_date=dt.date.today(),
            end_date=dt.date.today() + dt.timedelta(days=int(365 * maturity_years)),
            day_count=self.expiry_date.day_count
        )
        
        protected_amount = notional * protection_level
        self.add_zero_coupon_bond(face_value=protected_amount, maturity=maturity)
        
        # 2. Ajouter une position longue sur des calls pour la participation
        if cap is None:
            # Sans plafond: utiliser des calls standards
            call_option = self.create_option(strike=current_price, is_call=True)
            quantity = participation_rate * notional / current_price
            self.options_portfolio.add_option(call_option, quantity)
            
            self.components.append({
                'type': 'participation_call',
                'object': call_option,
                'quantity': quantity
            })
        else:
            # Avec plafond: utiliser un call spread
            call_lower = self.create_option(strike=current_price, is_call=True)
            call_upper = self.create_option(strike=current_price * (1 + cap), is_call=True)
            
            quantity = participation_rate * notional / current_price
            self.options_portfolio.add_option(call_lower, quantity)
            self.options_portfolio.add_option(call_upper, -quantity)
            
            self.components.append({
                'type': 'capped_participation',
                'lower_strike': current_price,
                'upper_strike': current_price * (1 + cap),
                'quantity': quantity
            })
        
        # Nom du produit avec détails pertinents
        product_name = f"Capital Protected Note {protection_level*100:.0f}% - {participation_rate*100:.0f}% Part."
        if cap:
            product_name += f" (capped at {cap*100:.0f}%)"
        
        # Calculer le prix total du produit
        total_price = self._calculate_total_price()
        
        # Ajouter le produit au portefeuille de produits structurés
        self.structured_portfolio.add_structured_product(
            product_name=product_name,
            product_type="capital_protected_note",
            components=self.components.copy(),
            price=total_price,
            quantity=1.0  # Par défaut une unité
        )
    
    def reverse_convertible(self, 
                          coupon_rate: float, 
                          barrier_level: float,
                          notional: float = 100.0) -> None:
        """
        Crée un reverse convertible (obligation convertible inversée).
        
        Args:
            coupon_rate: Taux du coupon annuel
            barrier_level: Niveau de la barrière (en % du prix spot)
            notional: Montant nominal de l'investissement
        """
        current_price = self.market_data.prix_spot
        maturity_years = self.expiry_date.maturity_years
        
        # Réinitialiser les composants pour le nouveau produit
        self.components = []
        
        # 1. Ajouter l'obligation zéro-coupon pour le paiement du coupon et du principal
        maturity = Maturity(
            start_date=dt.date.today(),
            end_date=dt.date.today() + dt.timedelta(days=int(365 * maturity_years)),
            day_count=self.expiry_date.day_count
        )
        
        # Paiement total à maturité (nominal + coupon)
        total_payment = notional * (1 + coupon_rate * maturity_years)
        self.add_zero_coupon_bond(face_value=total_payment, maturity=maturity)
        
        # 2. Ajouter un put down-and-in (le risque baissier pour l'investisseur)
        barrier = current_price * barrier_level
        strike = current_price
        quantity = -notional / current_price  # Position courte (vente)
        
        self.add_put_down_in(strike=strike, barrier_level=barrier, quantity=quantity)
        
        # Nom du produit avec détails pertinents
        product_name = f"Reverse Convertible {coupon_rate*100:.2f}% - {barrier_level*100:.0f}% Barrier"
        
        # Calculer le prix total du produit
        total_price = self._calculate_total_price()
        
        # Ajouter le produit au portefeuille de produits structurés
        self.structured_portfolio.add_structured_product(
            product_name=product_name,
            product_type="reverse_convertible",
            components=self.components.copy(),
            price=total_price,
            quantity=1.0  # Par défaut une unité
        )
    
    def barrier_digital(self, 
                      barrier_level: float, 
                      payout: float,
                      is_up_direction: bool = True,
                      is_knock_in: bool = True,
                      notional: float = 100.0) -> None:
        """
        Crée une option digitale à barrière.
        
        Args:
            barrier_level: Niveau de la barrière (en valeur absolue)
            payout: Montant du paiement si l'option est activée et dans la monnaie
            is_up_direction: True pour une barrière up, False pour down
            is_knock_in: True pour knock-in, False pour knock-out
            notional: Montant nominal de l'investissement
        """
        current_price = self.market_data.prix_spot
        
        # Réinitialiser les composants pour le nouveau produit
        self.components = []
        
        # Créer l'option avec barrière
        barrier_type = TypeBarriere.knock_in if is_knock_in else TypeBarriere.knock_out
        barrier_direction = DirectionBarriere.up if is_up_direction else DirectionBarriere.down
        
        # Déterminer le type d'option digitale en fonction de la direction de la barrière
        if is_up_direction:
            # Pour une barrière up, utiliser une option digitale call
            # avec un strike proche de la barrière
            digital_strike = barrier_level * 0.99
            self.add_digital_call(
                strike=digital_strike,
                payout=payout,
                quantity=notional / payout
            )
        else:
            # Pour une barrière down, utiliser une option digitale put
            # avec un strike proche de la barrière
            digital_strike = barrier_level * 1.01
            self.add_digital_put(
                strike=digital_strike,
                payout=payout,
                quantity=notional / payout
            )
            
        # Ajouter la composante barrière
        barrier_option = self.create_option(
            strike=current_price,
            is_call=is_up_direction,  # call pour up, put pour down
            barrier_type=barrier_type,
            barrier_direction=barrier_direction,
            barrier_level=barrier_level
        )
        
        # Nous allons simplement stocker cette option dans les composants
        # mais ne pas l'ajouter au portefeuille car son payoff est déjà
        # capturé par l'option digitale
        self.components.append({
            'type': 'barrier',
            'object': barrier_option
        })
        
        # Construction du nom du produit avec des détails pertinents
        barrier_dir = "Up" if is_up_direction else "Down"
        barrier_type = "Knock-In" if is_knock_in else "Knock-Out"
        barrier_pct = barrier_level / current_price * 100
        
        product_name = f"{barrier_dir} {barrier_type} Digital - {barrier_pct:.0f}% - {payout:.2f} Payout"
        
        # Calculer le prix total du produit
        total_price = self._calculate_total_price()
        
        # Ajouter le produit au portefeuille de produits structurés
        self.structured_portfolio.add_structured_product(
            product_name=product_name,
            product_type="barrier_digital",
            components=self.components.copy(),
            price=total_price,
            quantity=1.0  # Par défaut une unité
        )
    
    def _calculate_total_price(self) -> float:
        """
        Calcule le prix total du produit structuré en fonction de ses composants.
        
        Returns:
            float: Prix total du produit
        """
        total_price = 0
        for component in self.components:
            if component.get('price'):
                # Si le prix est directement disponible
                total_price += component['price']
            elif component.get('object') and hasattr(component['object'], 'price'):
                # Si nous avons un objet avec un attribut price
                total_price += component['object'].price
            elif component.get('type') == 'zero_coupon_bond' and component.get('object'):
                # Pour les obligations zéro-coupon
                total_price += component['object'].price
        
        return total_price
    
    def create_strategy(self, product_name: str, params: dict, quantity: float = 1.0) -> None:
        """
        Crée un produit structuré prédéfini en fonction du nom du produit
        et des paramètres fournis.
        
        Args:
            product_name: Nom du type de produit à créer
            params: Dictionnaire contenant les paramètres nécessaires à la création du produit
            quantity: Quantité du produit à ajouter au portefeuille
        
        Returns:
            None
        """
        product_name = product_name.lower()
        
        # Récupérer les paramètres communs
        notional = params.get("notional", 100.0)
        
        # Appeler la méthode correspondante avec les bons paramètres
        if product_name == "capital protected note":
            self.capital_protected_note(
                protection_level=params.get("protection_level", 1.0),
                participation_rate=params.get("participation_rate", 1.0),
                notional=notional,
                cap=params.get("cap", None)
            )
        elif product_name == "reverse convertible":
            self.reverse_convertible(
                coupon_rate=params.get("coupon_rate", 0.05),
                barrier_level=params.get("barrier_level", 0.8),
                notional=notional
            )
        elif product_name == "barrier digital":
            self.barrier_digital(
                barrier_level=params.get("barrier_level", 1.1),
                payout=params.get("payout", 10.0),
                is_up_direction=params.get("is_up_direction", True),
                is_knock_in=params.get("is_knock_in", True),
                notional=notional
            )
        else:
            raise ValueError(f"Type de produit structuré non reconnu: {product_name}")
        
        # Mettre à jour la quantité si nécessaire
        if quantity != 1.0:
            # Récupérer le dernier produit ajouté
            last_product_id = len(self.structured_portfolio.products) - 1
            if last_product_id >= 0:
                self.structured_portfolio.products[last_product_id]['quantity'] = quantity