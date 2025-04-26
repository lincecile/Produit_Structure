import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Optional
from src.options.Pricing_option.Classes_Both.module_barriere import TypeBarriere, DirectionBarriere
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_LSM import LSM_method
from src.options.Pricing_option.Classes_Both.module_option import Option
from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche  
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_brownian import Brownian
from src.options.Pricing_option.Classes_Both.derivatives import OptionDerivatives
import plotly.graph_objects as go
import pandas as pd
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
        price, _, _ = LSM_method(option).LSM(self.brownian, self.market_data)

        self.options.append({
            'type': 'Call' if option.call else "Put",
            'barriere': (option.barriere.type_barriere.value + ' - ' + option.barriere.direction_barriere.value) if option.barriere and option.barriere.type_barriere and option.barriere.direction_barriere else np.nan,
            'niveau barriere': round(option.barriere.niveau_barriere,2) if option.barriere and option.barriere.niveau_barriere else np.nan,
            'strike': option.prix_exercice,
            'quantity': quantity,
            'premium': price,
        })
        
        self.option_objects.append(option)

        combined = defaultdict(lambda: {'quantity': 0})

        for item in self.options:
            rounded_premium = round(float(item['premium']), 2) 
            key = (item['type'], item['barriere'], item['niveau barriere'], item['strike'], rounded_premium)
            combined[key]['quantity'] += item['quantity']

        # Résultat formaté
        result = [
            {'type': k[0], 'barriere': k[1], 'niveau barriere': k[2], 'strike': k[3], 'premium': k[4], 'quantity': v['quantity']}
            for k, v in combined.items()
        ]

        self.options = result
        print(self.options)
    
    def clear_portfolio(self):
        """Vide le portefeuille d'options."""
        self.options.clear()
        self.option_objects.clear()
    
    def remove_option_quantity(self, option_index: int, quantity_to_remove: float) -> bool:
        """
        Supprime une quantité spécifique d'une option existante dans le portefeuille.
        
        Args:
            option_index: Indice de l'option dans le portefeuille
            quantity_to_remove: Quantité à supprimer (valeur positive)
            
        Returns:
            bool: True si l'opération a réussi, False sinon
        """
        if option_index < 0 or option_index >= len(self.options):
            print(f"Erreur: Index d'option invalide ({option_index}). Le portefeuille contient {len(self.options)} options.")
                
        current_quantity = self.options[option_index]['quantity']
        
        # Si on supprime plus que la quantité disponible
        if (abs(quantity_to_remove) > abs(current_quantity)) or (abs(quantity_to_remove) == abs(current_quantity)):
            del self.options[option_index]
            del self.option_objects[option_index]
        else:
            # Sinon, on réduit la quantité
            # On garde le signe de la position (long/short)
            sign = 1 if current_quantity > 0 else -1
            self.options[option_index]['quantity'] -= quantity_to_remove * sign
            print(self.options)


    def price_portfolio(self) -> float:
        """Calcule le prix total du portefeuille"""
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
        greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        portfolio_greeks =  {greek: sum(self.calculate_option_greeks(i, pricer_options)[greek] for i in range(len(self.options))) for greek in greeks}
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
    
    def get_portfolio_detail(self) -> Dict:
        
        if not self.options:
            print("Aucune option dans le portefeuille.")
            return pd.DataFrame() 

        df = pd.DataFrame(self.options)
        df.index.name = "Numéro de l'option"
        
        # Ajout d'une colonne pour la valeur totale de la position
        df['valeur_position'] = (df['quantity'] * df['premium']).round(4)
        
        # Renommer les colonnes pour plus de clarté
        df = df.rename(columns={
            'type': 'Type',
            'strike': 'Strike',
            'quantity': 'Quantité',
            'premium': 'Prime payée/reçue',
        })
        return df

    def compute_payoff(self, option: Option, spot_prices: np.ndarray, quantity: float, premium: float, show_premium: bool = True):
        """Calcule le payoff d'une option avec prise en compte de la barrière."""
        is_call = option.call
        payoff = np.maximum(0, (spot_prices - option.prix_exercice) if is_call else (option.prix_exercice - spot_prices))
        
        # Gestion de la barrière
        if option.barriere:
            niveau = option.barriere.niveau_barriere

            if option.barriere.type_barriere is TypeBarriere.knock_in:
                if option.barriere.direction_barriere is DirectionBarriere.up:
                    payoff = np.where(spot_prices >= niveau, payoff, 0)
                elif option.barriere.direction_barriere is DirectionBarriere.down:
                    payoff = np.where(spot_prices <= niveau, payoff, 0)
            
            elif option.barriere.type_barriere is TypeBarriere.knock_out:
                if option.barriere.direction_barriere is DirectionBarriere.up:
                    payoff = np.where(spot_prices >= niveau, 0, payoff)
                elif option.barriere.direction_barriere is DirectionBarriere.down:
                    payoff = np.where(spot_prices <= niveau, 0, payoff)

        # Prendre en compte la quantité
        payoff *= quantity
        
        # Ajuster la prime si demandé
        if show_premium:
            payoff -= quantity * premium
            
        return payoff

    def plot_portfolio_payoff(self, price_range: float = 0.3, num_points: int = 1000, 
                  show_individual: bool = True, show_premium: bool = True):
        """
        Trace le graphique du payoff total du portefeuille ainsi que les payoffs individuels
        en utilisant Plotly, optimisé pour Streamlit.
        """
        
        if not self.options:
            raise ValueError("Le portefeuille est vide")
        
        # Créer une plage de prix
        current_price = self.market_data.prix_spot
        spot_prices = np.linspace(0, current_price*2, num_points)
        
        # Calculer le payoff total
        total_payoff = np.zeros_like(spot_prices)

        # Créer une figure Plotly
        fig = go.Figure()
        
        for i, option_info in enumerate(self.options):
            option = self.option_objects[i]
            quantity = option_info['quantity']
            premium = option_info['premium']

            payoff = self.compute_payoff(option, spot_prices, quantity, premium, show_premium=show_premium)
            
            # Plot individuel

            if show_individual:
                bar = option_info['barriere'] if option_info['barriere'] is not np.nan else ' '
                level = 'niveau = ' + str(option_info['niveau barriere']) if option_info['barriere'] is not np.nan else ' '
                titre = f"{option_info['type']} {bar} {level} K = {option_info['strike']:.2f} (x{quantity})"
                fig.add_trace(go.Scatter(
                    x=spot_prices,
                    y=payoff,
                    mode='lines',
                    line=dict(dash='dash', width=1.5),
                    opacity=0.6,
                    name=titre
                ))
            
            total_payoff += payoff
        
        # Tracer le payoff total
        fig.add_trace(go.Scatter(
            x=spot_prices,
            y=total_payoff,
            mode='lines',
            line=dict(color='blue', width=3),
            name='Stratégie complète'
        ))
        
        # Ajouter la ligne horizontale à zéro
        fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="grey")
        
        # Ajouter une ligne verticale pour le prix actuel
        fig.add_vline(x=current_price, line_width=1.5, line_dash="dash", line_color="green",
                    annotation_text=f"Prix actuel S={current_price:.2f}")
        
        # Configurer la mise en page - optimisé pour Streamlit
        title = "Profit/Perte du portefeuille d'options" + (" (primes incluses)" if show_premium else "")
        fig.update_layout(
            title=title,
            xaxis_title="Prix du sous-jacent",
            yaxis_title="Profit/Perte",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            autosize=True,
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
        return fig
       

    def plot_option_payoff(self, option_index: int, price_range: float = 0.3, 
                        num_points: int = 1000, show_premium: bool = True):
        """
        Trace le graphique du payoff d'une option spécifique en utilisant Plotly,
        optimisé pour Streamlit.
        
        Args:
            option_index: Indice de l'option dans le portefeuille
            price_range: Plage de variation du prix en pourcentage autour du prix actuel (ex: 0.3 = ±30%)
            num_points: Nombre de points pour le calcul du payoff
            show_premium: Afficher le coût de la prime dans le payoff
            
        Returns:
            fig: Figure Plotly à afficher dans Streamlit
        """
        import plotly.graph_objects as go
        
        if option_index >= len(self.options):
            raise ValueError(f"Index d'option invalide: {option_index}, le portefeuille contient {len(self.options)} options")
        
        option_info = self.options[option_index]
        strike = option_info['strike']
        quantity = option_info['quantity']
        premium = option_info['premium']
        is_call = option_info['type'].lower() == 'call'
        
        # Créer une plage de prix autour du strike
        current_price = self.market_data.prix_spot
        spot_prices = np.linspace(0, current_price*2, num_points)
        
        # Calculer le payoff avec et sans prime
        payoff_without_premium = quantity * np.maximum(0, (spot_prices - strike) if is_call else (strike - spot_prices))
        payoff_with_premium = payoff_without_premium - quantity * premium
        
        # Créer figure Plotly directement avec go.Figure()
        fig = go.Figure()
        
        if show_premium:
            fig.add_trace(go.Scatter(
                x=spot_prices,
                y=payoff_with_premium,
                mode='lines',
                line=dict(width=2),
                name=f"{option_info['type']} K={strike:.2f} (avec prime)"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=spot_prices,
                y=payoff_without_premium,
                mode='lines',
                line=dict(width=2),
                name=f"{option_info['type']} K={strike:.2f} (sans prime)"
            ))
        
        # Ajouter les lignes de référence
        # Ligne horizontale à zéro
        fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="grey")
        
        # Ligne verticale pour le strike
        fig.add_vline(x=strike, line_width=1.5, line_dash="dash", line_color="grey",
                    annotation_text=f"Strike K={strike:.2f}")
        
        # Ligne verticale pour le prix actuel
        fig.add_vline(x=current_price, line_width=1.5, line_dash="dash", line_color="green",
                    annotation_text=f"Prix actuel S={current_price:.2f}")
        
        # Configurer la mise en page - optimisé pour Streamlit
        fig.update_layout(
            title=f"Payoff pour {option_info['type']} (K={strike:.2f}, Quantité={quantity})",
            xaxis_title="Prix du sous-jacent",
            yaxis_title="Profit/Perte",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            autosize=True,
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig