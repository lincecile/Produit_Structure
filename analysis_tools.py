import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Callable, Union
import time

class AnalysisTools:
    """
    Classe permettant d'effectuer diverses analyses sur des options:
    - Matrices de P&L
    - Tests de stress
    - Analyses de sensibilité
    """
    
    def __init__(self, option, pricer, params=None):
        """
        Initialisation de l'outil d'analyse
        
        Args:
            option: L'option à analyser (EuropeanOption, AsianOption, etc.)
            pricer: Le pricer à utiliser (BlackScholesPricer, SemiAnalyticalPricer, MonteCarloPricer)
            params: Les paramètres du modèle (HestonParameters ou autre)
        """
        self.option = option
        self.pricer = pricer
        self.params = params
        self.initial_option_state = self._save_option_state()
        self.initial_params_state = self._save_params_state() if params else None
        
    def _save_option_state(self) -> Dict[str, Any]:
        """Sauvegarde l'état actuel de l'option"""
        return {
            'spot_price': self.option.spot_price,
            'strike': self.option.strike,
            'maturity': self.option.maturity,
            'risk_free_rate': self.option.risk_free_rate,
            'is_call': self.option.is_call
        }
    
    def _save_params_state(self) -> Dict[str, Any]:
        """Sauvegarde l'état actuel des paramètres du modèle"""
        if hasattr(self.params, 'kappa'):  # Pour HestonParameters
            return {
                'kappa': self.params.kappa,
                'theta': self.params.theta,
                'v0': self.params.v0,
                'sigma': self.params.sigma,
                'rho': self.params.rho
            }
        else:  # Pour d'autres types de paramètres
            return vars(self.params).copy()
    
    def _restore_initial_state(self):
        """Restaure l'option et les paramètres à leur état initial"""
        for key, value in self.initial_option_state.items():
            setattr(self.option, key, value)
        
        if self.params and self.initial_params_state:
            for key, value in self.initial_params_state.items():
                setattr(self.params, key, value)
    
    def compute_pnl_matrix(self, 
                          spot_range: List[float], 
                          vol_range: List[float] = None,
                          param_name: str = 'v0',
                          days_to_expiry: int = None) -> pd.DataFrame:
        """
        Calcule une matrice de P&L pour différents niveaux de prix spot et de volatilité
        
        Args:
            spot_range: Liste des prix spot à tester
            vol_range: Liste des volatilités à tester (v0 pour Heston)
            param_name: Nom du paramètre de volatilité à modifier ('v0' pour Heston)
            days_to_expiry: Nombre de jours avant expiration pour le calcul du P&L
            
        Returns:
            DataFrame contenant la matrice de P&L
        """
        # Si days_to_expiry est spécifié, on ajuste la maturité
        original_maturity = self.option.maturity
        if days_to_expiry is not None:
            remaining_maturity = days_to_expiry / 365.0
            self.option.maturity = remaining_maturity
        
        # Prix initial de l'option (valeur théorique à t=0)
        initial_price = self.pricer.price()
        
        # Créer la matrice de résultats
        if vol_range is None:
            # Si on ne varie que le spot price
            results = []
            for spot in spot_range:
                self.option.spot_price = spot
                new_price = self.pricer.price()
                pnl = new_price - initial_price
                results.append({'Spot': spot, 'P&L': pnl})
            
            df = pd.DataFrame(results)
            
            # Restaurer l'état initial
            self._restore_initial_state()
            return df
        else:
            # Si on varie spot et volatilité
            results = np.zeros((len(spot_range), len(vol_range)))
            
            for i, spot in enumerate(spot_range):
                self.option.spot_price = spot
                
                for j, vol in enumerate(vol_range):
                    # Mettre à jour le paramètre de volatilité
                    if hasattr(self.params, param_name):
                        setattr(self.params, param_name, vol)
                    elif param_name == 'volatility' and hasattr(self.pricer, 'volatility'):
                        self.pricer.volatility = vol
                    
                    new_price = self.pricer.price()
                    pnl = new_price - initial_price
                    results[i, j] = pnl
            
            # Restaurer l'état initial
            self._restore_initial_state()
            if days_to_expiry is not None:
                self.option.maturity = original_maturity
                
            # Créer un DataFrame
            df = pd.DataFrame(results, index=spot_range, columns=vol_range)
            df.index.name = 'Spot'
            df.columns.name = 'Volatilité'
            
            return df
    
    def plot_pnl_matrix(self, pnl_matrix: pd.DataFrame, title: str = 'Matrice de P&L'):
        """
        Affiche une matrice de P&L sous forme de heatmap
        
        Args:
            pnl_matrix: DataFrame contenant la matrice de P&L
            title: Titre du graphique
        """
        plt.figure(figsize=(12, 8))
        
        if isinstance(pnl_matrix.columns, pd.RangeIndex):
            # Cas simple: juste une série de P&L par spot
            plt.plot(pnl_matrix['Spot'], pnl_matrix['P&L'], 'b-', linewidth=2)
            plt.xlabel('Prix spot')
            plt.ylabel('P&L')
            plt.title(title)
            plt.grid(True)
        else:
            # Cas matriciel: heatmap
            heatmap = plt.pcolormesh(
                np.array(pnl_matrix.columns), 
                np.array(pnl_matrix.index), 
                pnl_matrix.values, 
                cmap='RdYlGn', 
                shading='auto'
            )
            
            plt.colorbar(heatmap, label='P&L')
            plt.xlabel(pnl_matrix.columns.name)
            plt.ylabel(pnl_matrix.index.name)
            plt.title(title)
        
        plt.tight_layout()
        return plt.gcf()  # Return the figure for further customization or saving
    
    def stress_test(self, 
                   scenario_params: List[Dict[str, Any]], 
                   scenario_names: List[str] = None) -> pd.DataFrame:
        """
        Effectue des tests de stress selon différents scénarios de marché
        
        Args:
            scenario_params: Liste de dictionnaires contenant les paramètres pour chaque scénario
                             Chaque dict peut contenir des clés comme 'spot_price', 'v0', 'kappa', etc.
            scenario_names: Liste des noms des scénarios (optionnel)
            
        Returns:
            DataFrame contenant les résultats des tests de stress
        """
        results = []
        
        # Prix de référence avec les paramètres actuels
        base_price = self.pricer.price()
        
        for i, scenario in enumerate(scenario_params):
            # Appliquer les changements de paramètres pour ce scénario
            for param_name, param_value in scenario.items():
                if hasattr(self.option, param_name):
                    setattr(self.option, param_name, param_value)
                elif self.params and hasattr(self.params, param_name):
                    setattr(self.params, param_name, param_value)
                elif hasattr(self.pricer, param_name):
                    setattr(self.pricer, param_name, param_value)
            
            # Calculer le nouveau prix
            try:
                start_time = time.time()
                stress_price = self.pricer.price()
                calc_time = time.time() - start_time
                
                # Calculer le P&L absolu et relatif
                pnl_abs = stress_price - base_price
                pnl_rel = pnl_abs / base_price * 100 if base_price != 0 else float('inf')
                
                # Créer un dictionnaire de résultats
                result = {
                    'Scenario': scenario_names[i] if scenario_names and i < len(scenario_names) else f"Scenario {i+1}",
                    'Prix': stress_price,
                    'P&L Absolu': pnl_abs,
                    'P&L Relatif (%)': pnl_rel,
                    'Temps de calcul (s)': calc_time
                }
                
                # Ajouter les paramètres du scénario au résultat
                for param_name, param_value in scenario.items():
                    result[param_name] = param_value
                
                results.append(result)
            
            except Exception as e:
                print(f"Erreur dans le scénario {i+1}: {str(e)}")
                
            # Restaurer l'état initial après chaque scénario
            self._restore_initial_state()
        
        # Créer un DataFrame
        df = pd.DataFrame(results)
        
        # Réorganiser les colonnes pour une meilleure lisibilité
        cols = ['Scenario', 'Prix', 'P&L Absolu', 'P&L Relatif (%)', 'Temps de calcul (s)']
        other_cols = [col for col in df.columns if col not in cols]
        df = df[cols + other_cols]
        
        return df
    
    def historical_scenario_test(self, 
                                historical_data: Dict[str, Dict[str, float]], 
                                returns_calculation: bool = True) -> pd.DataFrame:
        """
        Effectue des tests de stress basés sur des scénarios historiques
        
        Args:
            historical_data: Dictionnaire des scénarios historiques
                             Format: {'Nom du scénario': {'spot_price': X, 'volatility': Y, ...}}
            returns_calculation: Si True, calcule les rendements au lieu des prix absolus
            
        Returns:
            DataFrame contenant les résultats des tests de stress historiques
        """
        scenarios = []
        scenario_names = []
        
        # Convertir les données historiques en scénarios
        for scenario_name, params in historical_data.items():
            scenarios.append(params)
            scenario_names.append(scenario_name)
            
        # Utiliser la méthode stress_test existante
        results = self.stress_test(scenarios, scenario_names)
        
        # Si on veut des rendements relatifs plutôt que des prix absolus
        if returns_calculation:
            base_price = self.pricer.price()
            results['Rendement (%)'] = (results['Prix'] / base_price - 1) * 100
            
        return results
    
    def sensitivity_analysis(self, 
                            param_name: str, 
                            param_range: List[float], 
                            calc_greeks: bool = False) -> pd.DataFrame:
        """
        Effectue une analyse de sensibilité sur un paramètre spécifique
        
        Args:
            param_name: Nom du paramètre à analyser ('spot_price', 'strike', 'v0', etc.)
            param_range: Liste des valeurs du paramètre à tester
            calc_greeks: Si True, calcule les Greeks associés (si applicable)
            
        Returns:
            DataFrame contenant les résultats de l'analyse de sensibilité
        """
        results = []
        base_price = self.pricer.price()
        
        for value in param_range:
            # Mettre à jour le paramètre
            if hasattr(self.option, param_name):
                setattr(self.option, param_name, value)
            elif self.params and hasattr(self.params, param_name):
                setattr(self.params, param_name, value)
            elif hasattr(self.pricer, param_name):
                setattr(self.pricer, param_name, value)
            else:
                raise ValueError(f"Le paramètre '{param_name}' n'existe pas dans l'option, les paramètres ou le pricer.")
            
            # Calculer le nouveau prix
            new_price = self.pricer.price()
            
            # Calculer les Greeks si demandé
            greeks = {}
            if calc_greeks:
                if hasattr(self.pricer, 'first_order_derivative'):
                    if param_name == 'spot_price':
                        greeks['Delta'] = self.pricer.first_order_derivative('spot_price', 0.01)
                        greeks['Gamma'] = self.pricer.second_order_derivative('spot_price', 0.01)
                    elif param_name == 'maturity':
                        greeks['Theta'] = -self.pricer.first_order_derivative('maturity', 0.01)
                    elif param_name in ('v0', 'volatility'):
                        greeks['Vega'] = self.pricer.first_order_derivative(param_name, 0.01) * 0.01
                    elif param_name == 'risk_free_rate':
                        greeks['Rho'] = self.pricer.first_order_derivative('risk_free_rate', 0.01) * 0.01
            
            # Créer un dictionnaire de résultats
            result = {
                param_name: value,
                'Prix': new_price,
                'Différence': new_price - base_price,
                'Variation (%)': (new_price - base_price) / base_price * 100 if base_price != 0 else float('inf')
            }
            
            # Ajouter les Greeks s'ils ont été calculés
            result.update(greeks)
            
            results.append(result)
        
        # Restaurer l'état initial
        self._restore_initial_state()
        
        # Créer un DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def plot_sensitivity(self, 
                        sensitivity_df: pd.DataFrame, 
                        x_column: str = None, 
                        y_column: str = 'Prix', 
                        title: str = None):
        """
        Affiche les résultats d'une analyse de sensibilité
        
        Args:
            sensitivity_df: DataFrame contenant les résultats de l'analyse de sensibilité
            x_column: Nom de la colonne à utiliser pour l'axe X (défaut: première colonne)
            y_column: Nom de la colonne à utiliser pour l'axe Y (défaut: 'Prix')
            title: Titre du graphique
        """
        if x_column is None:
            x_column = sensitivity_df.columns[0]
            
        if title is None:
            title = f"Sensibilité du {y_column} au {x_column}"
            
        plt.figure(figsize=(10, 6))
        plt.plot(sensitivity_df[x_column], sensitivity_df[y_column], 'b-o', linewidth=2)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()  # Return the figure for further customization or saving