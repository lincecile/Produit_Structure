import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Callable, Union
import time
import copy

class AnalysisTools:
    """
    Classe générique permettant d'effectuer diverses analyses sur des options:
    - Matrices de P&L
    - Tests de stress
    - Analyses de sensibilité
    
    Compatible avec n'importe quel pricer qui possède une méthode price()
    """
    
    def __init__(self, option, pricer, params=None):
        """
        Initialisation de l'outil d'analyse
        
        Args:
            option: L'option à analyser (doit avoir une structure d'attributs)
            pricer: Le pricer à utiliser (doit avoir une méthode price())
            params: Les paramètres du modèle (optionnel)
        """
        self.option = option
        self.pricer = pricer
        self.params = params
        # Copier l'état initial des objets pour pouvoir les restaurer
        self._initial_option = copy.deepcopy(option)
        self._initial_params = copy.deepcopy(params) if params else None
    
    def _restore_initial_state(self):
        """Restaure l'option et les paramètres à leur état initial"""
        # Restaurer tous les attributs de l'option
        for attr, value in vars(self._initial_option).items():
            setattr(self.option, attr, value)
        
        # Restaurer tous les attributs des paramètres si présents
        if self.params and self._initial_params:
            for attr, value in vars(self._initial_params).items():
                setattr(self.params, attr, value)
    
    def compute_pnl_matrix(self, 
                          x_param_name: str,
                          x_param_values: List[float], 
                          y_param_name: str = None,
                          y_param_values: List[float] = None) -> pd.DataFrame:
        """
        Calcule une matrice de P&L pour différentes valeurs de paramètres
        
        Args:
            x_param_name: Nom du paramètre pour l'axe X (ex: 'spot_price')
            x_param_values: Liste des valeurs à tester pour ce paramètre
            y_param_name: Nom du paramètre pour l'axe Y (optionnel)
            y_param_values: Liste des valeurs à tester pour le paramètre Y
            
        Returns:
            DataFrame contenant la matrice de P&L
        """
        # Prix initial de l'option (valeur théorique à t=0)
        initial_price = self.pricer.price()
        
        # Si on varie un seul paramètre
        if y_param_name is None or y_param_values is None:
            results = []
            for x_value in x_param_values:
                # Mettre à jour le paramètre X
                self._set_param_value(x_param_name, x_value)
                
                # Calculer le nouveau prix
                new_price = self.pricer.price()
                pnl = new_price - initial_price
                
                results.append({
                    x_param_name: x_value,
                    'Price': new_price,
                    'P&L': pnl,
                    'P&L (%)': (pnl / initial_price * 100) if initial_price != 0 else float('inf')
                })
            
            # Créer un DataFrame
            df = pd.DataFrame(results)
            
        else:
            # Si on varie deux paramètres (matrice)
            matrix = np.zeros((len(x_param_values), len(y_param_values)))
            
            for i, x_value in enumerate(x_param_values):
                self._set_param_value(x_param_name, x_value)
                
                for j, y_value in enumerate(y_param_values):
                    self._set_param_value(y_param_name, y_value)
                    
                    # Calculer le nouveau prix
                    new_price = self.pricer.price()
                    pnl = new_price - initial_price
                    matrix[i, j] = pnl
            
            # Créer un DataFrame
            df = pd.DataFrame(matrix, 
                             index=x_param_values, 
                             columns=y_param_values)
            df.index.name = x_param_name
            df.columns.name = y_param_name
        
        # Restaurer l'état initial
        self._restore_initial_state()
        return df
    
    def _set_param_value(self, param_name, value):
        """
        Méthode utilitaire pour définir la valeur d'un paramètre
        Cherche le paramètre dans l'option, puis dans les paramètres, puis dans le pricer
        """
        # Vérifier si le paramètre est dans l'option
        if hasattr(self.option, param_name):
            setattr(self.option, param_name, value)
        # Sinon, vérifier s'il est dans les paramètres
        elif self.params and hasattr(self.params, param_name):
            setattr(self.params, param_name, value)
        # Sinon, vérifier s'il est dans le pricer
        elif hasattr(self.pricer, param_name):
            setattr(self.pricer, param_name, value)
        else:
            raise ValueError(f"Le paramètre '{param_name}' n'existe pas dans l'option, les paramètres ou le pricer.")
    
    def plot_pnl_matrix(self, pnl_matrix: pd.DataFrame, title: str = 'Matrice de P&L'):
        """
        Affiche une matrice de P&L sous forme de graphique
        
        Args:
            pnl_matrix: DataFrame contenant la matrice de P&L
            title: Titre du graphique
        """
        plt.figure(figsize=(12, 8))
        
        # Vérifier si le DataFrame est une matrice 2D ou un tableau 1D
        if isinstance(pnl_matrix.index, pd.RangeIndex) or len(pnl_matrix.columns) <= 4:
            # C'est un tableau 1D - faire un simple graphique en ligne
            x_col = pnl_matrix.columns[0]  # Première colonne = paramètre
            y_col = 'P&L' if 'P&L' in pnl_matrix.columns else 'Price' # attention erreur ici si matrice en 2D
            
            plt.plot(pnl_matrix[x_col], pnl_matrix[y_col], 'b-', linewidth=2, marker='o')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(title)
            plt.grid(True)
            
        else:
            # C'est une matrice 2D - faire une heatmap
            heatmap = plt.pcolormesh(
                pnl_matrix.columns.astype(float),
                pnl_matrix.index.astype(float),
                pnl_matrix.values,
                cmap='RdYlGn',
                shading='auto'
            )
            
            plt.colorbar(heatmap, label='P&L')
            plt.xlabel(pnl_matrix.columns.name)
            plt.ylabel(pnl_matrix.index.name)
            plt.title(title)
        
        plt.tight_layout()
        return plt.gcf()
    
    def stress_test(self, 
                   scenario_params: List[Dict[str, Any]], 
                   scenario_names: List[str] = None) -> pd.DataFrame:
        """
        Effectue des tests de stress selon différents scénarios
        
        Args:
            scenario_params: Liste de dictionnaires de paramètres pour chaque scénario
                             Format: [{'param1': value1, 'param2': value2, ...}, ...]
            scenario_names: Liste des noms des scénarios (optionnel)
            
        Returns:
            DataFrame contenant les résultats des tests de stress
        """
        results = []
        
        # Prix de référence avec les paramètres actuels
        base_price = self.pricer.price()
        
        for i, scenario in enumerate(scenario_params):
            # Appliquer tous les changements de paramètres pour ce scénario
            for param_name, param_value in scenario.items():
                self._set_param_value(param_name, param_value)
            
            # Calculer le nouveau prix
            try:
                start_time = time.time()
                stress_price = self.pricer.price()
                calc_time = time.time() - start_time
                
                # Calculer le P&L absolu et relatif
                pnl_abs = stress_price - base_price
                pnl_rel = (pnl_abs / base_price * 100) if base_price != 0 else float('inf')
                
                # Créer un dictionnaire de résultats
                result = {
                    'Scenario': scenario_names[i] if scenario_names and i < len(scenario_names) else f"Scenario {i+1}",
                    'Price': stress_price,
                    'P&L': pnl_abs,
                    'P&L (%)': pnl_rel,
                    'Calc Time (s)': calc_time
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
        main_cols = ['Scenario', 'Price', 'P&L', 'P&L (%)', 'Calc Time (s)']
        other_cols = [col for col in df.columns if col not in main_cols]
        df = df[
            [col for col in main_cols if col in df.columns] + 
            [col for col in other_cols if col in df.columns]
        ]
        
        return df