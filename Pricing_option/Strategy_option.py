import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List, Dict, Tuple, Union, Optional


class OptionsPortfolio:
    """
    Classe qui gère un portefeuille d'options (calls et puts) et calcule le payoff total
    ainsi que les grecques associées.
    """
    
    def __init__(self):
        """Initialise un portefeuille d'options vide."""
        self.options = []
        
    def add_option(self, option_type: str, strike: float, quantity: float = 1.0, 
                   premium: Optional[float] = None):
        """
        Ajoute une option au portefeuille.
        
        Args:
            option_type: Type d'option ('call' ou 'put')
            strike: Prix d'exercice
            quantity: Quantité/nombre de contrats (positif pour position longue, négatif pour position courte)
            premium: Prime payée/reçue par option (optionnel)
        """
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Le type d'option doit être 'call' ou 'put'")
        
        self.options.append({
            'type': option_type.lower(),
            'strike': strike,
            'quantity': quantity,
            'premium': premium
        })
    
    def clear_portfolio(self):
        """Vide le portefeuille d'options."""
        self.options = []
    
    def calculate_payoff(self, spot_prices: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calcule le payoff total du portefeuille pour un ou plusieurs prix spot.
        
        Args:
            spot_prices: Prix spot unique ou liste/array de prix spots
            
        Returns:
            Le payoff total du portefeuille pour chaque prix spot
        """
        # Conversion en numpy array pour faciliter les calculs vectoriels
        if isinstance(spot_prices, (int, float)):
            spot_prices = np.array([spot_prices])
        else:
            spot_prices = np.array(spot_prices)
        
        # Initialisation du payoff total à zéro
        total_payoff = np.zeros_like(spot_prices, dtype=float)
        
        # Calcul du payoff pour chaque option dans le portefeuille
        for option in self.options:
            if option['type'] == 'call':
                # Payoff d'un call: max(spot - strike, 0)
                option_payoff = np.maximum(spot_prices - option['strike'], 0)
            else:  # put
                # Payoff d'un put: max(strike - spot, 0)
                option_payoff = np.maximum(option['strike'] - spot_prices, 0)
            
            # Ajustement selon la quantité
            option_payoff *= option['quantity']
            
            # Soustraction de la prime si elle est spécifiée
            if option['premium'] is not None:
                option_payoff -= option['premium'] * option['quantity']
                
            # Ajout au payoff total
            total_payoff += option_payoff
            
        # Si un seul prix spot a été fourni, retourner un scalaire
        if len(total_payoff) == 1:
            return float(total_payoff[0])
        
        return total_payoff
    
    def _black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, 
                             option_type: str) -> float:
        """
        Calcule le prix d'une option selon le modèle de Black-Scholes.
        
        Args:
            S: Prix spot du sous-jacent
            K: Prix d'exercice
            T: Durée jusqu'à l'expiration (en années)
            r: Taux d'intérêt sans risque
            sigma: Volatilité annualisée
            option_type: Type d'option ('call' ou 'put')
            
        Returns:
            Le prix de l'option selon Black-Scholes
        """
        if T <= 0:
            # Pour les options à échéance immédiate, calculer la valeur intrinsèque
            if option_type == 'call':
                return max(S - K, 0)
            else:  # put
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        return price
    
    def _calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str) -> Dict[str, float]:
        """
        Calcule les grecques pour une option selon le modèle de Black-Scholes.
        
        Args:
            S: Prix spot du sous-jacent
            K: Prix d'exercice
            T: Durée jusqu'à l'expiration (en années)
            r: Taux d'intérêt sans risque
            sigma: Volatilité annualisée
            option_type: Type d'option ('call' ou 'put')
            
        Returns:
            Un dictionnaire contenant les valeurs des grecques
        """
        if T <= 0:
            # Pour les options à échéance immédiate, les grecques sont des cas spéciaux
            if option_type == 'call':
                delta = 1.0 if S > K else 0.0
            else:  # put
                delta = -1.0 if S < K else 0.0
            return {
                'delta': delta,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calcul du delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        # Calcul du gamma (identique pour call et put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Calcul du vega (identique pour call et put, en % de la valeur du sous-jacent)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # divisé par 100 pour avoir le vega pour 1% de variation de vol
        
        # Calcul du theta (en valeur par jour)
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = (term1 + term2) / 365  # divisé par 365 pour avoir le theta par jour
        else:  # put
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = (term1 + term2) / 365  # divisé par 365 pour avoir le theta par jour
            
        # Calcul du rho (en % du prix pour 1% de variation du taux)
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # divisé par 100 pour avoir le rho pour 1% de variation du taux
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
        
    def calculate_portfolio_greeks(self, S: float, T: float, r: float, sigma: float) -> Dict[str, float]:
        """
        Calcule les grecques pour l'ensemble du portefeuille d'options.
        
        Args:
            S: Prix spot du sous-jacent
            T: Durée jusqu'à l'expiration (en années)
            r: Taux d'intérêt sans risque
            sigma: Volatilité annualisée
            
        Returns:
            Un dictionnaire contenant les grecques agrégées du portefeuille
        """
        # Initialiser les grecques du portefeuille à zéro
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'price': 0.0  # Ajouter le prix total du portefeuille
        }
        
        # Calculer les grecques pour chaque option et les agréger
        for option in self.options:
            K = option['strike']
            option_type = option['type']
            quantity = option['quantity']
            
            # Calculer le prix de l'option
            option_price = self._black_scholes_price(S, K, T, r, sigma, option_type)
            portfolio_greeks['price'] += option_price * quantity
            
            # Calculer les grecques de l'option
            option_greeks = self._calculate_greeks(S, K, T, r, sigma, option_type)
            
            # Agréger les grecques en tenant compte de la quantité
            for greek, value in option_greeks.items():
                portfolio_greeks[greek] += value * quantity
                
        return portfolio_greeks
    
    def plot_greeks_vs_spot(self, T: float, r: float, sigma: float, 
                           spot_range=None, figsize=(15, 10)):
        """
        Trace l'évolution des grecques du portefeuille en fonction du prix spot.
        
        Args:
            T: Durée jusqu'à l'expiration (en années)
            r: Taux d'intérêt sans risque
            sigma: Volatilité annualisée
            spot_range: Plage de prix spot à utiliser. Si None, une plage sera automatiquement déterminée.
            figsize: Taille de la figure (largeur, hauteur) en pouces.
            
        Returns:
            La figure matplotlib et les axes
        """
        if not self.options:
            raise ValueError("Le portefeuille est vide, rien à tracer.")
        
        # Déterminer la plage de prix spot si non spécifiée
        if spot_range is None:
            strikes = [opt['strike'] for opt in self.options]
            min_strike, max_strike = min(strikes), max(strikes)
            padding = (max_strike - min_strike) * 0.5
            if padding < 10:  # Assurer un minimum de distance
                padding = 10
            spot_min = max(0, min_strike - padding)  # Éviter les valeurs négatives
            spot_max = max_strike + padding
            spot_range = np.linspace(spot_min, spot_max, 100)
        
        # Initialiser les listes pour stocker les valeurs des grecques
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        rhos = []
        prices = []
        
        # Calculer les grecques pour chaque prix spot
        for S in spot_range:
            greeks = self.calculate_portfolio_greeks(S, T, r, sigma)
            deltas.append(greeks['delta'])
            gammas.append(greeks['gamma'])
            thetas.append(greeks['theta'])
            vegas.append(greeks['vega'])
            rhos.append(greeks['rho'])
            prices.append(greeks['price'])
        
        # Créer la figure et les axes
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Tracer le prix
        axes[0, 0].plot(spot_range, prices, 'b-', linewidth=2)
        axes[0, 0].set_title('Prix du Portefeuille')
        axes[0, 0].set_xlabel('Prix Spot')
        axes[0, 0].set_ylabel('Prix')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Tracer le delta
        axes[0, 1].plot(spot_range, deltas, 'r-', linewidth=2)
        axes[0, 1].set_title('Delta du Portefeuille')
        axes[0, 1].set_xlabel('Prix Spot')
        axes[0, 1].set_ylabel('Delta')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tracer le gamma
        axes[1, 0].plot(spot_range, gammas, 'g-', linewidth=2)
        axes[1, 0].set_title('Gamma du Portefeuille')
        axes[1, 0].set_xlabel('Prix Spot')
        axes[1, 0].set_ylabel('Gamma')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Tracer le theta
        axes[1, 1].plot(spot_range, thetas, 'm-', linewidth=2)
        axes[1, 1].set_title('Theta du Portefeuille (par jour)')
        axes[1, 1].set_xlabel('Prix Spot')
        axes[1, 1].set_ylabel('Theta')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Tracer le vega
        axes[2, 0].plot(spot_range, vegas, 'c-', linewidth=2)
        axes[2, 0].set_title('Vega du Portefeuille (pour 1% de variation de vol)')
        axes[2, 0].set_xlabel('Prix Spot')
        axes[2, 0].set_ylabel('Vega')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Tracer le rho
        axes[2, 1].plot(spot_range, rhos, 'y-', linewidth=2)
        axes[2, 1].set_title('Rho du Portefeuille (pour 1% de variation du taux)')
        axes[2, 1].set_xlabel('Prix Spot')
        axes[2, 1].set_ylabel('Rho')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Identifier la stratégie et l'afficher dans le titre global
        strategy = self._identify_strategy()
        if strategy != "Custom strategy":
            fig.suptitle(f"Grecques de la stratégie: {strategy}", fontsize=16)
        else:
            fig.suptitle("Grecques du Portefeuille d'Options", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig, axes
    
    def plot_payoff(self, spot_range=None, include_individual_options=False, figsize=(10, 6)):
        """
        Trace le payoff du portefeuille d'options.
        
        Args:
            spot_range: Plage de prix spot à utiliser pour le graphique. 
                        Si None, une plage sera automatiquement déterminée.
            include_individual_options: Si True, affiche également les payoffs individuels de chaque option.
            figsize: Taille de la figure (largeur, hauteur) en pouces.
        
        Returns:
            La figure matplotlib et l'axe
        """
        if not self.options:
            raise ValueError("Le portefeuille est vide, rien à tracer.")
        
        # Déterminer la plage de prix spot si non spécifiée
        if spot_range is None:
            strikes = [opt['strike'] for opt in self.options]
            min_strike, max_strike = min(strikes), max(strikes)
            padding = (max_strike - min_strike) * 0.5
            if padding < 10:  # Assurer un minimum de distance
                padding = 10
            spot_min = max(0, min_strike - padding)  # Éviter les valeurs négatives
            spot_max = max_strike + padding
            spot_range = np.linspace(spot_min, spot_max, 200)
        
        # Calculer le payoff total
        total_payoff = self.calculate_payoff(spot_range)
        
        # Créer la figure et l'axe
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tracer le payoff total
        ax.plot(spot_range, total_payoff, 'b-', linewidth=2, label='Payoff Total')
        
        # Tracer les payoffs individuels si demandé
        if include_individual_options:
            colors = ['r', 'g', 'm', 'c', 'y', 'k']  # Couleurs pour différencier les options
            
            for i, option in enumerate(self.options):
                # Initialiser le payoff à zéro
                option_payoff = np.zeros_like(spot_range)
                
                # Calculer le payoff pour cette option
                if option['type'] == 'call':
                    option_payoff = np.maximum(spot_range - option['strike'], 0)
                else:  # put
                    option_payoff = np.maximum(option['strike'] - spot_range, 0)
                
                # Ajuster selon la quantité
                option_payoff *= option['quantity']
                
                # Soustraire la prime si spécifiée
                if option['premium'] is not None:
                    option_payoff -= option['premium'] * option['quantity']
                
                # Tracer le payoff individuel
                color = colors[i % len(colors)]
                position = "Long" if option['quantity'] > 0 else "Short"
                label = f"{position} {option['type'].capitalize()} (K={option['strike']})"
                ax.plot(spot_range, option_payoff, color=color, linestyle='--', 
                        alpha=0.7, label=label)
        
        # Ajouter une ligne horizontale à y=0
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Styliser le graphique
        ax.set_title('Payoff du Portefeuille d\'Options')
        ax.set_xlabel('Prix Spot du Sous-jacent')
        ax.set_ylabel('Profit/Perte')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Identifier la stratégie et l'afficher dans le titre
        strategy = self._identify_strategy()
        if strategy != "Custom strategy":
            plt.title(f"Payoff: {strategy}")
        
        return fig, ax
        
    def get_portfolio_summary(self) -> Dict:
        """
        Retourne un résumé du portefeuille d'options.
        
        Returns:
            Un dictionnaire contenant des informations sur le portefeuille
        """
        n_calls = sum(1 for opt in self.options if opt['type'] == 'call')
        n_puts = sum(1 for opt in self.options if opt['type'] == 'put')
        
        # Calcul des positions nettes
        net_call_position = sum(opt['quantity'] for opt in self.options if opt['type'] == 'call')
        net_put_position = sum(opt['quantity'] for opt in self.options if opt['type'] == 'put')
        
        # Calcul du coût total du portefeuille (primes)
        total_cost = sum(opt['quantity'] * opt['premium'] for opt in self.options 
                         if opt['premium'] is not None)
        
        # Identification des stratégies optionnelles communes
        strategy = self._identify_strategy()
        
        return {
            'n_options': len(self.options),
            'n_calls': n_calls,
            'n_puts': n_puts,
            'net_call_position': net_call_position,
            'net_put_position': net_put_position,
            'total_cost': total_cost,
            'identified_strategy': strategy
        }
    
    def _identify_strategy(self) -> str:
        """
        Tente d'identifier la stratégie optionnelle utilisée dans le portefeuille.
        
        Returns:
            Le nom de la stratégie identifiée ou 'Custom strategy' si non reconnue
        """
        # Simplification: cette méthode est un exemple et ne détecte que quelques stratégies de base
        # Dans une implémentation complète, cette logique serait plus élaborée
        
        if len(self.options) == 0:
            return "Empty portfolio"
        
        if len(self.options) == 1:
            opt = self.options[0]
            position = "Long" if opt['quantity'] > 0 else "Short"
            return f"{position} {opt['type'].capitalize()}"
        
        if len(self.options) == 2:
            # Vérifier s'il s'agit d'un spread
            if all(opt['type'] == 'call' for opt in self.options):
                # Les deux options sont des calls
                if sum(opt['quantity'] for opt in self.options) == 0:
                    return "Call Spread"
            
            if all(opt['type'] == 'put' for opt in self.options):
                # Les deux options sont des puts
                if sum(opt['quantity'] for opt in self.options) == 0:
                    return "Put Spread"
            
            # Vérifier s'il s'agit d'un straddle
            if (self.options[0]['type'] != self.options[1]['type'] and 
                self.options[0]['strike'] == self.options[1]['strike']):
                return "Straddle"
        
        if len(self.options) == 4:
            # Vérifier s'il s'agit d'un butterfly
            call_options = [opt for opt in self.options if opt['type'] == 'call']
            if len(call_options) == 3:
                strikes = sorted([opt['strike'] for opt in call_options])
                quantities = [opt['quantity'] for opt in self.options if opt['type'] == 'call']
                if len(strikes) == 3 and strikes[1] - strikes[0] == strikes[2] - strikes[1]:
                    if quantities.count(1) == 2 and quantities.count(-2) == 1:
                        return "Butterfly"
        
        # Si aucune stratégie n'est identifiée
        return "Custom strategy"
    
# Exemple d'utilisation
if __name__ == "__main__":
    # Création d'un portefeuille
    portfolio = OptionsPortfolio()
    
    # # Butterfly
    # portfolio.add_option('call', 90, 1, 10)    # Long call bas strike
    # portfolio.add_option('call', 100, -2, 5)   # Short 2 calls strike moyen
    # portfolio.add_option('call', 110, 1, 2)    # Long call haut strike

    # Butterfly
    portfolio.add_option('put', 80, -6*3, 0)    # Long call bas strike
    portfolio.add_option('put', 70, 5*3, 0)   # Short 2 calls strike moyen
    # portfolio.add_option('put', 100, 1, 0)    # Long call haut strike
    
    # Paramètres de marché
    S = 100  # Prix spot actuel
    T = 0.25  # Échéance (3 mois)
    r = 0.03  # Taux sans risque
    sigma = 0.2  # Volatilité
    
    # Calculer les grecques pour le prix spot actuel
    greeks = portfolio.calculate_portfolio_greeks(S, T, r, sigma)
    print("Grecques du portefeuille pour S =", S)
    for greek, value in greeks.items():
        print(f"{greek.capitalize()}: {value:.6f}")
    
    # Tracer les grecques en fonction du prix spot
    # fig, axes = portfolio.plot_greeks_vs_spot(T, r, sigma)
    # plt.tight_layout()
    # plt.show()
    
    # Tracer également le payoff
    fig2, ax2 = portfolio.plot_payoff(include_individual_options=True)
    plt.tight_layout()
    plt.show()