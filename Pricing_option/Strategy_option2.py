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
    portfolio.add_option(option, 1)    # Long call bas strike
    # portfolio.add_option(option, -2, price)   # Short 2 calls strike moyen
    portfolio.add_option(option, 2)    # Long call haut strike
    
    # Calcul et affichage des grecques du portefeuille
    portfolio_greeks = portfolio.calculate_portfolio_greeks()
    print(portfolio_greeks)
    print("Grecques du portefeuille:")
    for greek, value in portfolio_greeks.items():
        print(f"{greek.capitalize()}: {value:.6f}")
    
    # Obtenir le résumé du portefeuille
    summary = portfolio.get_portfolio_summary()
    print("\nRésumé du portefeuille:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Tracer le payoff
    # fig_payoff, ax_payoff = portfolio.plot_payoff(include_individual_options=True)
    
    # # Tracer les grecques (réduit le nombre de points pour accélérer)
    # spot_range = np.linspace(80, 120, 10)
    # fig_greeks, axes_greeks = portfolio.plot_greeks_vs_spot(spot_range=spot_range)
    
    # plt.show()