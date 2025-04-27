import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Union, Optional
from src.products import Product
from src.options.Pricing_option.Classes_Both.module_barriere import TypeBarriere, DirectionBarriere
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_LSM import LSM_method
from src.options.Pricing_option.Classes_Both.module_option import Option
from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche  
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_brownian import Brownian
from src.options.Pricing_option.Classes_Both.derivatives import OptionDerivatives
import plotly.graph_objects as go
import pandas as pd

class StructuredProductsPortfolio(Product):
    """
    Extension du portefeuille d'options pour inclure des produits structurés.
    Cette classe permet de gérer ensemble des options et des produits structurés.
    """
    
    def __init__(self, name: str,
                 price_history: Optional[np.array] = None,
                 price: Optional[float] = None,
                 volatility: Optional[float] = None) -> None:
        """
        Initialise un portefeuille de produits structurés vide.
        
        Args:
            name: Nom du portefeuille
            price_history: Historique des prix (optionnel)
            price: Prix actuel (optionnel)
            volatility: Volatilité (optionnel)
        """
        self.name = name
        self.price_history = price_history
        self.price = price or 0.0
        self.volatility = volatility
        
        self.structured_products = []  # initialize the list of structured products
    
    def add_structured_product(self, product_name: str, product_type: str, components: list, 
                              price: float, quantity: float = 1.0):
        """
        Ajoute un produit structuré au portefeuille.
        
        Args:
            product_name: Nom du produit (ex: "Capital Protected Note 90%")
            product_type: Type du produit (ex: "capital_protected_note", "reverse_convertible")
            components: Liste des composants du produit
            price: Prix total du produit
            quantity: Quantité de produits à ajouter
        """
        self.structured_products.append({
            'name': product_name,
            'type': product_type,
            'components': components,
            'price': price,
            'quantity': quantity,
            'value': price * quantity
        })
        
        # Mettre à jour le prix du portefeuille
        self.price = self._calculate_portfolio_price()

    def _calculate_portfolio_price(self):
        total = 0.0
        for product in self.structured_products:
            total += product['price'] * product['quantity']
        return total

    def clear_portfolio(self) -> None:
        """
        Vide le portefeuille de produits structurés.
        """
        self.structured_products.clear()
        self.price = 0.0
        
    def remove_product(self, product_index: int, quantity_to_remove: float) -> bool:
        """
        Supprime un produit du portefeuille.
        
        Args:
            product_index: Indice du produit dans le portefeuille
            
        Returns:
            bool: True si l'opération a réussi, False sinon
        """
        if product_index < 0 or product_index >= len(self.structured_products):
            print(f"Erreur: Index de produit invalide ({product_index}). Le portefeuille contient {len(self.structured_products)} produits.")
        
        current_quantity = self.structured_products[product_index]['quantity']
        
        if (abs(quantity_to_remove) > abs(current_quantity)) or (abs(quantity_to_remove) == abs(current_quantity)):
            del self.structured_products[product_index]
        else:
            # Sinon, on réduit la quantité
            # On garde le signe de la position (long/short)
            sign = 1 if current_quantity > 0 else -1
            self.structured_products[product_index]['quantity'] -= quantity_to_remove * sign
        
        # Mettre à jour le prix total du portefeuille
        self.price = self._get_price()

    def _get_price(self) -> float:
        """Calcule le prix total du portefeuille"""
        return sum(product['price'] * product['quantity'] for product in self.structured_products)
    
    def get_portfolio_summary(self) -> Dict:
        """
        Retourne un résumé du portefeuille de produits structurés.
        
        Returns:
            Dict: Un dictionnaire contenant des informations résumées sur le portefeuille
        """
        product_types  = Counter()
        total_value = sum(product['quantity'] * product['price'] for product in self.structured_products)

        for product in self.structured_products:
            product_types[product['type']] += product['quantity']

        return {
            'n_products': len(self.structured_products),
            'product_types': dict(product_types),
            'total_value': total_value,
            'portfolio_price': self.price
        }
    
    def get_portfolio_detail(self) -> pd.DataFrame:
        """
        Retourne les détails du portefeuille sous forme de DataFrame pandas.
        
        Returns:
            pd.DataFrame: Un DataFrame contenant les détails de chaque produit structuré
        """
        if not self.structured_products:
            print("Aucun produit structuré dans le portefeuille.")
            return pd.DataFrame()
        
        products_data = []
        for i, product in enumerate(self.structured_products):
            # Compter les types de composants
            component_counts = defaultdict(int)
            for comp in product['components']:
                component_counts[comp['type']] += 1
            
            products_data.append({
                'Numéro': i,
                'Nom': product['name'],
                'Type': product['type'],
                'Prix unitaire': round(product['price'], 2),
                'Quantité': product['quantity'],
                'Valeur totale': round(product['price'] * product['quantity'], 2),
                'Composants': ', '.join([f"{count} {comp_type}" for comp_type, count in component_counts.items()])
            })
        
        return pd.DataFrame(products_data)
    
    def compute_payoff(self, product_index: int, spot_prices: np.ndarray) -> np.ndarray:
        """
        Calcule le payoff à maturité d'un produit structuré pour une gamme de prix du sous-jacent.
        
        Args:
            product_index: Indice du produit dans le portefeuille
            spot_prices: Tableau numpy contenant une gamme de prix du sous-jacent
            
        Returns:
            np.ndarray: Tableau numpy contenant le payoff pour chaque prix du sous-jacent
        """
        if product_index < 0 or product_index >= len(self.structured_products):
            raise ValueError(f"Index de produit invalide: {product_index}")
        
        product = self.structured_products[product_index]
        product_type = product['type']
        components = product['components']
        quantity = product['quantity']
        
        # Calculer le payoff en fonction du type de produit
        if product_type == "capital_protected_note":
            return self._compute_capital_protected_note_payoff(components, spot_prices) * quantity
        elif product_type == "reverse_convertible":
            return self._compute_reverse_convertible_payoff(components, spot_prices) * quantity
        elif product_type == "barrier_digital":
            return self._compute_barrier_digital_payoff(components, spot_prices) * quantity
        elif product_type == "athena_autocall":
            return self._compute_athena_autocall_payoff(components, spot_prices) * quantity
        else:
            # Si le type n'est pas reconnu, on renvoie un payoff nul
            print(f"Type de produit non reconnu pour le calcul du payoff: {product_type}")
            return np.zeros_like(spot_prices)
        
    def _compute_capital_protected_note_payoff(self, components: List[Dict], spot_prices: np.ndarray) -> np.ndarray:
        """
        Calcule le payoff d'une note à capital protégé.
        
        Args:
            components: Liste des composants du produit
            spot_prices: Tableau numpy contenant une gamme de prix du sous-jacent
            
        Returns:
            np.ndarray: Tableau numpy contenant le payoff pour chaque prix du sous-jacent
        """
        payoff = np.zeros_like(spot_prices)
        
        # Trouver l'obligation zéro-coupon
        bond_component = next((comp for comp in components if comp['type'] == 'zero_coupon_bond'), None)
        if bond_component and 'object' in bond_component:
            # Ajouter la valeur nominale de l'obligation
            payoff += bond_component['object'].face_value
        
        # Trouver les composants de participation
        call_component = next((comp for comp in components if comp['type'] == 'participation_call'), None)
        if call_component and 'object' in call_component and 'quantity' in call_component:
            option = call_component['object']
            quantity = call_component['quantity']
            
            # Ajouter le payoff du call
            option_payoff = np.maximum(0, spot_prices - option.prix_exercice)
            payoff += option_payoff * quantity
        
        # Vérifier s'il s'agit d'une participation plafonnée
        capped_component = next((comp for comp in components if comp['type'] == 'capped_participation'), None)
        if capped_component:
            lower_strike = capped_component['lower_strike']
            upper_strike = capped_component['upper_strike']
            quantity = capped_component['quantity']
            
            # Calculer le payoff plafonné
            payoffs = np.minimum(upper_strike - lower_strike, np.maximum(0, spot_prices - lower_strike))
            payoff += payoffs * quantity
        
        return payoff
    
    def _compute_reverse_convertible_payoff(self, components: List[Dict], spot_prices: np.ndarray) -> np.ndarray:
        """
        Calcule le payoff d'un reverse convertible.
        
        Args:
            components: Liste des composants du produit
            spot_prices: Tableau numpy contenant une gamme de prix du sous-jacent
            
        Returns:
            np.ndarray: Tableau numpy contenant le payoff pour chaque prix du sous-jacent
        """
        payoff = np.zeros_like(spot_prices)
        
        # Trouver l'obligation zéro-coupon
        bond_component = next((comp for comp in components if comp['type'] == 'zero_coupon_bond'), None)
        if bond_component and 'object' in bond_component:
            # Ajouter la valeur nominale de l'obligation
            payoff += bond_component['object'].face_value
        
        # Trouver le put down-in
        put_component = next((comp for comp in components if comp['type'] == 'put_down_in'), None)
        if put_component and 'object' in put_component and 'quantity' in put_component:
            option = put_component['object']
            quantity = put_component['quantity']
            barrier = option.barriere.niveau_barriere
            strike = option.prix_exercice
            
            # Payoff du put à barrière (down-in)
            # Si le prix descend sous la barrière, le put est activé
            put_payoff = np.maximum(0, strike - spot_prices)
            barriere_activee = spot_prices <= barrier
            put_payoff = np.where(barriere_activee, put_payoff, 0)
            
            payoff += put_payoff * quantity
        
        return payoff
    
    def _compute_barrier_digital_payoff(self, components: List[Dict], spot_prices: np.ndarray) -> np.ndarray:
        """
        Calcule le payoff d'une option digitale à barrière.
        
        Args:
            components: Liste des composants du produit
            spot_prices: Tableau numpy contenant une gamme de prix du sous-jacent
            
        Returns:
            np.ndarray: Tableau numpy contenant le payoff pour chaque prix du sous-jacent
        """
        payoff = np.zeros_like(spot_prices)
        
        # Trouver le composant digital_call ou digital_put
        digital_call = next((comp for comp in components if comp['type'] == 'digital_call'), None)
        digital_put = next((comp for comp in components if comp['type'] == 'digital_put'), None)
        barrier_comp = next((comp for comp in components if comp['type'] == 'barrier'), None)
        
        if digital_call:
            strike = digital_call['strike']
            payout = digital_call['payout']
            quantity = digital_call['quantity']
            
            # Paiement digital: payout si spot > strike, 0 sinon
            digital_payoff = np.where(spot_prices > strike, payout, 0)
            payoff += digital_payoff * quantity
            
        elif digital_put:
            strike = digital_put['strike']
            payout = digital_put['payout']
            quantity = digital_put['quantity']
            
            # Paiement digital: payout si spot < strike, 0 sinon
            digital_payoff = np.where(spot_prices < strike, payout, 0)
            payoff += digital_payoff * quantity
            
        # Appliquer la condition de barrière si présente
        if barrier_comp and 'object' in barrier_comp:
            barrier_option = barrier_comp['object']
            barrier_level = barrier_option.barriere.niveau_barriere
            is_knock_in = barrier_option.barriere.type_barriere == TypeBarriere.knock_in
            is_up = barrier_option.barriere.direction_barriere == DirectionBarriere.up
            
            if is_knock_in:
                # Pour knock-in: payoff actif seulement si la barrière est touchée
                if is_up:
                    payoff = np.where(spot_prices >= barrier_level, payoff, 0)
                else:  # down
                    payoff = np.where(spot_prices <= barrier_level, payoff, 0)
            else:  # knock-out
                # Pour knock-out: payoff nul si la barrière est touchée
                if is_up:
                    payoff = np.where(spot_prices >= barrier_level, 0, payoff)
                else:  # down
                    payoff = np.where(spot_prices <= barrier_level, 0, payoff)
        
        return payoff

    def _compute_athena_autocall_payoff(self, components: List[Dict], spot_prices: np.ndarray) -> np.ndarray:
        """
        Calcule le payoff d'un produit Athena autocall à maturité, avec prise en compte
        de l'effet mémoire si présent.
        
        Args:
            components: Liste des composants du produit
            spot_prices: Tableau numpy contenant une gamme de prix du sous-jacent
            
        Returns:
            np.ndarray: Tableau numpy contenant le payoff pour chaque prix du sous-jacent
        """
        payoff = np.zeros_like(spot_prices)
        
        # Trouver l'obligation zéro-coupon (pour le remboursement du nominal)
        bond_component = next((comp for comp in components if comp['type'] == 'zero_coupon_bond'), None)
        
        # Valeur nominale à rembourser
        notional = bond_component['object'].face_value if bond_component and 'object' in bond_component else 100.0
        
        # Trouver les observations autocall (dernier niveau de barrière)
        autocall_observations = [comp for comp in components if comp['type'] == 'autocall_observation']
        
        if autocall_observations:
            # Prendre la dernière observation (maturité)
            final_observation = autocall_observations[-1]
            final_barrier = final_observation['barrier_level']
            
            # Utiliser le coupon total (qui inclut l'effet mémoire) si disponible
            if 'total_coupon' in final_observation:
                final_coupon = final_observation['total_coupon']
            else:
                final_coupon = final_observation['coupon_rate']
            
            # Si le prix à maturité est au-dessus de la dernière barrière,
            # payer le nominal + coupon final (avec effet mémoire si applicable)
            payoff = np.where(spot_prices >= final_barrier, 
                            notional * (1 + final_coupon), 
                            notional)  # Par défaut, rembourser le nominal
        
        # Trouver le put down-in (pour la protection à la baisse)
        put_component = next((comp for comp in components if comp['type'] == 'put_down_in'), None)
        
        if put_component and 'object' in put_component and 'quantity' in put_component:
            option = put_component['object']
            quantity = put_component['quantity']
            barrier = option.barriere.niveau_barriere
            strike = option.prix_exercice
            
            # Payoff du put à barrière (down-in)
            # Si le prix descend sous la barrière, le put est activé et le capital n'est plus protégé
            put_payoff = np.maximum(0, strike - spot_prices)
            barriere_activee = spot_prices <= barrier
            put_payoff = np.where(barriere_activee, put_payoff, 0)
            
            # Soustraire la perte du put du remboursement du nominal
            # (La quantité est négative car nous sommes courts sur le put)
            payoff += put_payoff * quantity
        
        return payoff
    
    def plot_product_payoff(self, product_index: int, price_range: float = 0.3, 
                          num_points: int = 1000, show_premium: bool = True):
        """
        Trace le graphique du payoff d'un produit structuré spécifique en utilisant Plotly.
        
        Args:
            product_index: Indice du produit dans le portefeuille
            price_range: Plage de variation du prix en pourcentage autour du prix actuel
            num_points: Nombre de points pour le calcul du payoff
            show_premium: Afficher le coût du produit dans le payoff
            
        Returns:
            fig: Figure Plotly à afficher
        """
        import plotly.graph_objects as go
        
        if product_index >= len(self.structured_products):
            raise ValueError(f"Index de produit invalide: {product_index}, le portefeuille contient {len(self.structured_products)} produits")
        
        product = self.structured_products[product_index]
        product_name = product['name']
        price = product['price']
        
        # Déterminer l'étendue du prix spot
        # Pour simplifier, on prend une plage autour du prix actuel
        # En pratique, il faudrait utiliser le prix du sous-jacent du produit
        current_price = 100  # Prix arbitraire, à remplacer par le prix réel
        if 'components' in product and len(product['components']) > 0:
            # Essayer de trouver le prix d'exercice d'une option pour centrer la plage
            for comp in product['components']:
                if 'object' in comp and hasattr(comp['object'], 'prix_exercice'):
                    current_price = comp['object'].prix_exercice
                    break
        
        min_price = 0
        max_price = current_price * 2
        spot_prices = np.linspace(min_price, max_price, num_points)
        
        # Calculer le payoff
        payoff = self.compute_payoff(product_index, spot_prices)
        
        # Soustraire le prix du produit si demandé
        if show_premium:
            payoff -= price * product['quantity']
        
        # Créer la figure Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=spot_prices,
            y=payoff,
            mode='lines',
            line=dict(width=2),
            name=product_name
        ))
        
        # Ajouter les lignes de référence
        # Ligne horizontale à zéro
        fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="grey")
        
        # Ligne verticale pour le prix actuel
        fig.add_vline(x=current_price, line_width=1.5, line_dash="dash", line_color="green",
                     annotation_text=f"Prix de référence")
        
        # Configurer la mise en page
        title_suffix = " (coût inclus)" if show_premium else ""
        fig.update_layout(
            title=f"Payoff à maturité: {product_name}{title_suffix}",
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
    
    def plot_portfolio_payoff(self, price_range: float = 0.3, num_points: int = 1000, 
                            show_individual: bool = True, show_premium: bool = True):
        """
        Trace le graphique du payoff total du portefeuille de produits structurés
        en utilisant Plotly.
        
        Args:
            price_range: Plage de variation du prix en pourcentage autour du prix actuel
            num_points: Nombre de points pour le calcul du payoff
            show_individual: Afficher les payoffs individuels des produits
            show_premium: Afficher le coût des produits dans le payoff
            
        Returns:
            fig: Figure Plotly à afficher
        """
        import plotly.graph_objects as go
        
        if not self.structured_products:
            raise ValueError("Le portefeuille est vide")
        
        # Déterminer l'étendue du prix spot (comme avant)
        current_price = 100  # Prix arbitraire, à remplacer par le prix réel
        min_price = 0
        max_price = current_price * 2
        spot_prices = np.linspace(min_price, max_price, num_points)
        
        # Calculer le payoff total
        total_payoff = np.zeros_like(spot_prices)
        
        # Créer la figure Plotly
        fig = go.Figure()
        
        for i, product in enumerate(self.structured_products):
            payoff = self.compute_payoff(i, spot_prices)
            
            # Soustraire le prix du produit si demandé
            if show_premium:
                payoff -= product['price'] * product['quantity']
            
            # Plot individuel
            if show_individual:
                fig.add_trace(go.Scatter(
                    x=spot_prices,
                    y=payoff,
                    mode='lines',
                    line=dict(dash='dash', width=1.5),
                    opacity=0.6,
                    name=product['name']
                ))
            
            total_payoff += payoff
        
        # Tracer le payoff total
        fig.add_trace(go.Scatter(
            x=spot_prices,
            y=total_payoff,
            mode='lines',
            line=dict(color='blue', width=3),
            name='Portefeuille complet'
        ))
        
        # Ajouter la ligne horizontale à zéro
        fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="grey")
        
        # Ajouter une ligne verticale pour le prix actuel
        fig.add_vline(x=current_price, line_width=1.5, line_dash="dash", line_color="green",
                     annotation_text=f"Prix de référence")
        
        # Configurer la mise en page
        title_suffix = " (coûts inclus)" if show_premium else ""
        fig.update_layout(
            title=f"Profit/Perte du portefeuille de produits structurés{title_suffix}",
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
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig