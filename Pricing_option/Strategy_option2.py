import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List, Dict, Tuple, Union, Optional
from Classes_MonteCarlo_LSM.module_LSM import LSM_method
from Classes_Both.module_option import Option
from Classes_Both.module_marche import DonneeMarche  
from Classes_MonteCarlo_LSM.module_brownian import Brownian
import datetime as dt
from Classes_Both.derivatives import OptionDerivatives


class OptionsPortfolio:
    """
    Classe qui gère un portefeuille d'options (calls et puts) et calcule le payoff total
    ainsi que les grecques associées en utilisant les méthodes LSM et derivatives.
    """
    
    def __init__(self, brownian: Brownian, market: DonneeMarche) -> None:
        """Initialise un portefeuille d'options vide."""
        self.options = []
        self.option_objects = []  # Pour stocker les objets Option
        self.market_data = market   # Pour stocker les données de marché
        self.brownian = brownian      # Pour stocker l'objet Brownian
        
    def add_option(self, option: Option, quantity: float = 1.0):
        """
        Ajoute une option au portefeuille.
        Args:
            option: l'objet Option à ajouter
            quantity: Quantité/nombre de contrats (positif pour position longue, négatif pour position courte)
            premium: Prime payée/reçue par option (optionnel)
        """
        price, std_error, intevalles = LSM_method(option).LSM(self.brownian, self.market_data)
        
        self.options.append({
            'type': 'Call' if option.call else "Put",
            'strike': option.prix_exercice,
            'quantity': quantity,
            'premium': price,
        })
        
        self.option_objects.append(option)
    
    def clear_portfolio(self):
        """Vide le portefeuille d'options."""
        self.options.clear()
        self.option_objects.clear()
    
    def price_portfolio(self) -> float:
        """
        Calcule le prix total du portefeuille
        """
        return sum(opt['premium'] * opt['quantity'] for opt in self.options)
    
    def calculate_option_greeks(self, option_index: int, pricer_options: dict = None) -> Dict[str, float]:
        """
        Calcule les grecques pour une option spécifique en utilisant la classe OptionDerivatives.
        
        Args:
            option_index: Indice de l'option dans le portefeuille
            pricer_options: Options supplémentaires pour le pricer
            
        Returns:
            Un dictionnaire contenant les valeurs des grecques
        """
        if not self.market_data or not self.brownian:
            raise ValueError("Les données de marché et les paramètres browniens doivent être définis avant le calcul des grecques")
        
        option = self.option_objects[option_index]
        quantity = self.options[option_index]['quantity']
       
        # Créer le pricer LSM
        pricer = LSM_method(option)
        
        # Initialiser le calculateur de grecques
        derivatives = OptionDerivatives(option, self.market_data, pricer, pricer_options if pricer_options else {})
        
        return {
            'delta': derivatives.delta(self.brownian) * quantity,
            'gamma': derivatives.gamma(self.brownian) * quantity,
            'theta': derivatives.theta(self.brownian) * quantity,
            'vega': derivatives.vega(self.brownian) * quantity,
            'rho': derivatives.rho(self.brownian) * quantity
        }
    
    def calculate_portfolio_greeks(self, pricer_options: dict = None) -> Dict[str, float]:
        """
        Calcule les grecques pour l'ensemble du portefeuille.
        Args:
            pricer_options: Options supplémentaires pour le pricer
        Returns:
            Un dictionnaire contenant les grecques agrégées du portefeuille
        """
        portfolio_greeks = {g: 0.0 for g in ['delta', 'gamma', 'theta', 'vega', 'rho']}
        # Calculer les grecques pour chaque option et les agréger
        for i in range(len(self.options)):
            option_greeks = self.calculate_option_greeks(i, pricer_options)
            
            # Agréger les grecques
            for greek, value in option_greeks.items():
                portfolio_greeks[greek] += value
                
        return portfolio_greeks
        
    def get_portfolio_summary(self) -> Dict:
        """
        Retourne un résumé du portefeuille d'options.
        """
        
        n_calls = sum(abs(opt['quantity']) for opt in self.options if opt['type'].lower() == 'call')
        n_puts = sum(abs(opt['quantity']) for opt in self.options if opt['type'].lower() == 'put')
        
        # Calcul des positions nettes
        net_call_position = sum(opt['quantity'] for opt in self.options if opt['type'].lower() == 'call')
        net_put_position = sum(opt['quantity'] for opt in self.options if opt['type'].lower() == 'put')
        
        # Calcul du coût total du portefeuille (primes)
        total_cost = sum(opt['quantity'] * opt['premium'] for opt in self.options)

        return {
            'n_options': n_calls + n_puts,
            'n_calls': n_calls,
            'n_puts': n_puts,
            'net_call_position': net_call_position,
            'net_put_position': net_put_position,
            'total_cost': total_cost,
            'portfolio_price': self.price_portfolio()
        }

    def _calculate_option_payoff(self, option_index: int, spot_prices: np.ndarray) -> np.ndarray:
        """
        Calcule le payoff d'une option pour une série de prix du sous-jacent.
        
        Args:
            option_index: Indice de l'option dans le portefeuille
            spot_prices: Array numpy des prix du sous-jacent
            
        Returns:
            Array numpy des payoffs correspondants
        """
        option_info = self.options[option_index]
        strike = option_info['strike']
        quantity = option_info['quantity']
        premium = option_info['premium']
        is_call = option_info['type'].lower() == 'call'
        
        if is_call:
            # Payoff d'un call: max(0, S - K)
            payoff = np.maximum(0, spot_prices - strike)
        else:
            # Payoff d'un put: max(0, K - S)
            payoff = np.maximum(0, strike - spot_prices)
        
        # Ajuster avec la quantité et la prime
        return quantity * (payoff - premium)
    
    def _calculate_total_payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Calcule le payoff total du portefeuille pour une série de prix du sous-jacent.
        
        Args:
            spot_prices: Array numpy des prix du sous-jacent
            
        Returns:
            Array numpy des payoffs totaux correspondants
        """
        total_payoff = np.zeros_like(spot_prices)
        
        for i in range(len(self.options)):
            total_payoff += self._calculate_option_payoff(i, spot_prices)
            
        return total_payoff
    
    def plot_option_payoff(self, option_index: int, price_range: float = 0.3, 
                           num_points: int = 1000, show_premium: bool = True) -> None:
        """
        Trace le graphique du payoff d'une option spécifique.
        
        Args:
            option_index: Indice de l'option dans le portefeuille
            price_range: Plage de variation du prix en pourcentage autour du prix actuel (ex: 0.3 = ±30%)
            num_points: Nombre de points pour le calcul du payoff
            show_premium: Afficher le coût de la prime dans le payoff
        """
        if option_index >= len(self.options):
            raise ValueError(f"Index d'option invalide: {option_index}, le portefeuille contient {len(self.options)} options")
        
        option_info = self.options[option_index]
        strike = option_info['strike']
        quantity = option_info['quantity']
        premium = option_info['premium']
        is_call = option_info['type'].lower() == 'call'
        
        # Créer une plage de prix autour du strike
        current_price = self.market_data.prix
        min_price = current_price * (1 - price_range)
        max_price = current_price * (1 + price_range)
        spot_prices = np.linspace(min_price, max_price, num_points)
        
        # Calculer le payoff avec et sans prime
        if is_call:
            payoff_without_premium = quantity * np.maximum(0, spot_prices - strike)
        else:
            payoff_without_premium = quantity * np.maximum(0, strike - spot_prices)
        
        payoff_with_premium = payoff_without_premium - quantity * premium
        
        # Tracer le graphique
        plt.figure(figsize=(10, 6))
        
        if show_premium:
            plt.plot(spot_prices, payoff_with_premium, label=f"{option_info['type']} K={strike:.2f} (avec prime)")
        else:
            plt.plot(spot_prices, payoff_without_premium, label=f"{option_info['type']} K={strike:.2f} (sans prime)")
        
        # Ajouter les lignes de référence
        plt.axhline(y=0, color='grey', linestyle='-', alpha=0.5)
        plt.axvline(x=strike, color='grey', linestyle='--', alpha=0.5, label=f"Strike K={strike:.2f}")
        plt.axvline(x=current_price, color='green', linestyle='--', alpha=0.5, label=f"Prix actuel S={current_price:.2f}")
        
        # Ajouter les labels et la légende
        plt.xlabel("Prix du sous-jacent")
        plt.ylabel("Profit/Perte")
        plt.title(f"Payoff pour {option_info['type']} (K={strike:.2f}, Quantité={quantity})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_portfolio_payoff(self, price_range: float = 0.3, num_points: int = 1000, 
                              show_individual: bool = True, show_premium: bool = True) -> None:
        """
        Trace le graphique du payoff total du portefeuille ainsi que les payoffs individuels si demandé.
        
        Args:
            price_range: Plage de variation du prix en pourcentage autour du prix actuel (ex: 0.3 = ±30%)
            num_points: Nombre de points pour le calcul du payoff
            show_individual: Afficher les payoffs individuels de chaque option
            show_premium: Afficher le coût des primes dans les payoffs
        """
        if not self.options:
            raise ValueError("Le portefeuille est vide")
        
        # Créer une plage de prix
        current_price = self.market_data.prix_spot
        min_price = current_price * (1 - price_range)
        max_price = current_price * (1 + price_range)
        spot_prices = np.linspace(min_price, max_price, num_points)
        
        # Calculer le payoff total
        total_payoff = np.zeros_like(spot_prices)
        
        plt.figure(figsize=(12, 8))
        
        # Tracer les payoffs individuels si demandé
        if show_individual:
            for i, option_info in enumerate(self.options):
                strike = option_info['strike']
                quantity = option_info['quantity']
                premium = option_info['premium']
                is_call = option_info['type'].lower() == 'call'
                
                if is_call:
                    payoff = quantity * np.maximum(0, spot_prices - strike)
                else:
                    payoff = quantity * np.maximum(0, strike - spot_prices)
                
                if show_premium:
                    payoff = payoff - quantity * premium
                
                plt.plot(spot_prices, payoff, linestyle='--', alpha=0.6, 
                         label=f"{option_info['type']} K={strike:.2f} (x{quantity})")
                
                total_payoff += payoff
        else:
            # Calculer le payoff total sans l'afficher individuellement
            for i in range(len(self.options)):
                option_info = self.options[i]
                strike = option_info['strike']
                quantity = option_info['quantity']
                premium = option_info['premium']
                is_call = option_info['type'].lower() == 'call'
                
                if is_call:
                    payoff = quantity * np.maximum(0, spot_prices - strike)
                else:
                    payoff = quantity * np.maximum(0, strike - spot_prices)
                
                if show_premium:
                    payoff = payoff - quantity * premium
                
                total_payoff += payoff
        
        # Tracer le payoff total
        plt.plot(spot_prices, total_payoff, linewidth=3, color='blue', label='Stratégie complète')
        
        # Ajouter les lignes de référence
        plt.axhline(y=0, color='grey', linestyle='-', alpha=0.5)
        plt.axvline(x=current_price, color='green', linestyle='--', alpha=0.7, 
                    label=f"Prix actuel S={current_price:.2f}")
        
        # Ajouter les strikes des options comme lignes verticales
        strikes = list(set([opt['strike'] for opt in self.options]))
        for strike in strikes:
            plt.axvline(x=strike, color='red', linestyle=':', alpha=0.5)
        
        # Ajouter les labels et la légende
        plt.xlabel("Prix du sous-jacent")
        plt.ylabel("Profit/Perte")
        title = "Profit/Perte du portefeuille d'options"
        if show_premium:
            title += " (primes incluses)"
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    
    def plot_strategy_analysis(self, price_range: float = 0.3, num_points: int = 1000) -> None:
        """
        Trace une analyse complète de la stratégie avec plusieurs graphiques:
        - Payoff total avec et sans primes
        - Points de break-even
        - Profit/perte maximal(e)
        
        Args:
            price_range: Plage de variation du prix en pourcentage autour du prix actuel (ex: 0.3 = ±30%)
            num_points: Nombre de points pour le calcul du payoff
        """
        if not self.options:
            raise ValueError("Le portefeuille est vide")
        
        # Créer une plage de prix
        current_price = self.market_data.prix
        min_price = current_price * (1 - price_range)
        max_price = current_price * (1 + price_range)
        spot_prices = np.linspace(min_price, max_price, num_points)
        
        # Calculer les payoffs avec et sans primes
        payoff_with_premium = np.zeros_like(spot_prices)
        payoff_without_premium = np.zeros_like(spot_prices)
        
        for option_info in self.options:
            strike = option_info['strike']
            quantity = option_info['quantity']
            premium = option_info['premium']
            is_call = option_info['type'].lower() == 'call'
            
            if is_call:
                option_payoff = quantity * np.maximum(0, spot_prices - strike)
            else:
                option_payoff = quantity * np.maximum(0, strike - spot_prices)
            
            payoff_without_premium += option_payoff
            payoff_with_premium += option_payoff - quantity * premium
        
        # Calculer les points de break-even (où payoff = 0)
        breakeven_points = []
        for i in range(1, len(spot_prices)):
            if (payoff_with_premium[i-1] < 0 and payoff_with_premium[i] >= 0) or \
               (payoff_with_premium[i-1] > 0 and payoff_with_premium[i] <= 0):
                # Interpolation linéaire pour trouver le point exact
                x1, x2 = spot_prices[i-1], spot_prices[i]
                y1, y2 = payoff_with_premium[i-1], payoff_with_premium[i]
                if y1 != y2:  # Éviter division par zéro
                    breakeven_x = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                    breakeven_points.append(breakeven_x)
        
        # Trouver le profit/perte max
        max_profit = np.max(payoff_with_premium)
        max_loss = np.min(payoff_with_premium)
        max_profit_price = spot_prices[np.argmax(payoff_with_premium)]
        max_loss_price = spot_prices[np.argmin(payoff_with_premium)]
        
        # Créer la figure avec deux sous-graphiques
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Graphique principal: Payoff total
        ax1.plot(spot_prices, payoff_without_premium, '--', color='grey', 
                 label='Sans primes', alpha=0.7)
        ax1.plot(spot_prices, payoff_with_premium, 'b-', linewidth=2, 
                 label='Avec primes')
        
        # Ajouter les lignes de référence et les points importants
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.axvline(x=current_price, color='green', linestyle='--', alpha=0.7, 
                   label=f"Prix actuel S={current_price:.2f}")
        
        # Ajouter les strikes comme lignes verticales
        strikes = list(set([opt['strike'] for opt in self.options]))
        for strike in strikes:
            ax1.axvline(x=strike, color='red', linestyle=':', alpha=0.5)
        
        # Marquer les points de break-even
        for bp in breakeven_points:
            ax1.plot(bp, 0, 'ro', markersize=8)
            ax1.text(bp, 0, f" BE: {bp:.2f}", verticalalignment='bottom')
        
        # Marquer les profits/pertes max
        ax1.plot(max_profit_price, max_profit, 'go', markersize=8)
        ax1.text(max_profit_price, max_profit, f" Max profit: {max_profit:.2f}", verticalalignment='bottom')
        
        ax1.plot(max_loss_price, max_loss, 'ro', markersize=8)
        ax1.text(max_loss_price, max_loss, f" Max loss: {max_loss:.2f}", verticalalignment='top')
        
        # Configurer le graphique principal
        ax1.set_xlabel("Prix du sous-jacent")
        ax1.set_ylabel("Profit/Perte")
        ax1.set_title("Analyse complète de la stratégie d'options")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Graphique secondaire: Répartition des options
        opt_types = []
        strikes_list = []
        quantities = []
        colors = []
        
        for opt in self.options:
            opt_types.append(opt['type'])
            strikes_list.append(opt['strike'])
            quantities.append(opt['quantity'])
            # Couleur: vert pour long, rouge pour short
            colors.append('green' if opt['quantity'] > 0 else 'red')
        
        # Barres pour les quantités d'options par strike
        ax2.bar(strikes_list, quantities, width=current_price*0.01, color=colors)
        
        # Ajouter les étiquettes
        for i, (strike, qty, opt_type) in enumerate(zip(strikes_list, quantities, opt_types)):
            direction = "Long" if qty > 0 else "Short"
            ax2.text(strike, qty, f"{direction} {opt_type}", 
                    ha='center', va='bottom' if qty > 0 else 'top', rotation=45)
        
        # Configurer le graphique secondaire
        ax2.set_xlabel("Prix d'exercice (Strike)")
        ax2.set_ylabel("Quantité")
        ax2.set_title("Composition du portefeuille d'options")
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Ajuster l'échelle des x pour qu'elle corresponde au graphique principal
        ax2.set_xlim([min_price, max_price])
        

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
        self.expiry_date = expiry_date or market_data.date_maturite
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

    def long_call(self, strike: float, quantity: float = 1.0, americaine: bool = True) -> None:
        """
        Ajoute un(des) call(s) long(s) au portefeuille.
        """
        if quantity <= 0:
            raise ValueError("La quantité doit être positive pour une position longue")
        
        option = self.create_option(strike, is_call=True, americaine=americaine)
        self.portfolio.add_option(option, quantity)
        
    def short_call(self, strike: float, quantity: float = 1.0, americaine: bool = True) -> None:
        """
        Ajoute un(des) call(s) short(s) au portefeuille.
        """
        if quantity <= 0:
            raise ValueError("La quantité doit être positive et sera convertie en position courte")
        
        option = self.create_option(strike, is_call=True, americaine=americaine)
        self.portfolio.add_option(option, -quantity)
    
    def long_put(self, strike: float, quantity: float = 1.0, americaine: bool = True) -> None:
        """
        Ajoute un(des) put(s) long(s) au portefeuille.
        """
        if quantity <= 0:
            raise ValueError("La quantité doit être positive pour une position longue")
        
        option = self.create_option(strike, is_call=False, americaine=americaine)
        self.portfolio.add_option(option, quantity)
    
    def short_put(self, strike: float, quantity: float = 1.0, americaine: bool = True) -> None:
        """
        Ajoute un(des) put(s) short(s) au portefeuille.
        """
        if quantity <= 0:
            raise ValueError("La quantité doit être positive et sera convertie en position courte")
        
        option = self.create_option(strike, is_call=False, americaine=americaine)
        self.portfolio.add_option(option, -quantity)

    def call_spread(self, lower_strike: float, upper_strike: float, quantity: float = 1.0, americaine: bool = True) -> None:
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
    
    def put_spread(self, lower_strike: float, upper_strike: float, quantity: float = 1.0, americaine: bool = True) -> None:
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
    
    def strangle(self, put_strike: float, call_strike: float, quantity: float = 1.0, americaine: bool = True) -> None:
        """
        Crée un strangle: achat d'un put OTM et d'un call OTM.
        
        Args:
            put_strike: Strike du put (doit être < prix actuel)
            call_strike: Strike du call (doit être > prix actuel)
            quantity: Nombre de strangles
            americaine: True pour options européennes, False pour américaines
        """
        current_price = self.market_data.prix
        
        if not (put_strike < current_price < call_strike):
            raise ValueError(f"Pour un strangle, le strike du put ({put_strike}) doit être inférieur au prix actuel ({current_price}) et le strike du call ({call_strike}) supérieur")
        
        self.long_put(put_strike, quantity, americaine)
        self.long_call(call_strike, quantity, americaine)
    
    def straddle(self, strike: float, quantity: float = 1.0, americaine: bool = True) -> None:
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
                  quantity: float = 1.0, is_call: bool = True, americaine: bool = True) -> None:
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
               option_quantity: float = None, americaine: bool = True) -> None:
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

    def create_strategy(self, strategy_name: str, params: dict) -> None:
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
        
        # Vérifier si tous les paramètres nécessaires sont fournis
        required_params = {
            "call_spread": ["lower_strike", "upper_strike", "quantity", "americaine"],
            "put_spread": ["lower_strike", "upper_strike", "quantity", "americaine"],
            "strangle": ["put_strike", "call_strike", "quantity", "americaine"],
            "straddle": ["strike", "quantity", "americaine"],
            "butterfly": ["lower_strike", "middle_strike", "upper_strike", "quantity", "is_call", "americaine"],
            "collar": ["stock_quantity", "put_strike", "call_strike", "option_quantity", "americaine"]
        }

        americaine = params.get("americaine", True)
        quantity = params.get("quantity", 1.0)

        # Appeler la méthode correspondante avec les bons paramètres
        if strategy_name == "call_spread":
            self.call_spread(
                lower_strike=params["strike1"],
                upper_strike=params["strike2"],
                quantity=quantity,
                americaine=americaine
            )
        elif strategy_name == "put_spread":
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
            option_quantity = params.get("option_quantity", None)
            self.collar(
                put_strike=params["strike1"],
                call_strike=params["strike2"],
                option_quantity=option_quantity,
                americaine=americaine
            )
    

# Exemple d'utilisation
if __name__ == "__main__":
    # Création d'un portefeuille
    
    
    date_pricing = dt.date(2024, 1, 13)
    spot = 100
    volatite = 0.2
    risk_free_rate = 0.02
    dividende_ex_date = dt.date.today()
    dividende_montant = 0
    maturite = dt.date(2024, 10, 23)
    strike = 101
    option_exercice = 'Européenne' 
    option_type = "Call"
    convention_base_calendaire = 365
    parametre_alpha = 3
    pruning = True
    epsilon_arbre = 1e-15
    nb_pas = 100
    nb_chemin = 100000
    seed_choice = 42
    antithetic_choice = False
    poly_degree = 2
    regress_method = "Polynomial"
    donnee_marche = DonneeMarche(date_pricing, spot, volatite, risk_free_rate, risk_free_rate, dividende_ex_date, dividende_montant)
    option = Option(maturite, strike, barriere=None, 
                    americaine=False if option_exercice == 'Européenne' else True, call=True if option_type == "Call" else False,
                    date_pricing=date_pricing)

    brownian = Brownian(time_to_maturity=(maturite-date_pricing).days / convention_base_calendaire, nb_step=nb_pas, nb_trajectoire=nb_chemin, seed=seed_choice)
    
    portfolio = OptionsPortfolio(brownian,donnee_marche)
    
    # Ajout d'options pour former un butterfly
    # portfolio.add_option(option, 1)    # Long call bas strike
    # # portfolio.add_option(option, -2, price)   # Short 2 calls strike moyen
    # portfolio.add_option(option, 2)    # Long call haut strike

    # Créer un gestionnaire de stratégies
    strategy = OptionsStrategy(portfolio, donnee_marche, expiry_date=maturite)

    # Ajouter un bull call spread
    # strategy.call_spread(lower_strike=100, upper_strike=110, quantity=1.0)

    # Ajouter un straddle
    strategy.straddle(strike=105, quantity=1.0)

    # Visualiser le payoff de la stratégie complète
    portfolio.plot_portfolio_payoff(show_individual=True)
    
    # Calcul et affichage des grecques du portefeuille
    # portfolio_greeks = portfolio.calculate_portfolio_greeks()
    # print(portfolio_greeks)
    # print("Grecques du portefeuille:")
    # for greek, value in portfolio_greeks.items():
    #     print(f"{greek.capitalize()}: {value:.6f}")
    
    # # Obtenir le résumé du portefeuille
    # summary = portfolio.get_portfolio_summary()
    # print("\nRésumé du portefeuille:")
    # for key, value in summary.items():
    #     print(f"{key}: {value}")
    
    # Tracer le payoff
    # fig_payoff, ax_payoff = portfolio.plot_payoff(include_individual_options=True)
    
    # # Tracer les grecques (réduit le nombre de points pour accélérer)
    # spot_range = np.linspace(80, 120, 10)
    # fig_greeks, axes_greeks = portfolio.plot_greeks_vs_spot(spot_range=spot_range)
    
    # plt.show()