import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Callable, Union
import time
import copy

class AnalysisTools:
    """
    Classe permettant différentes analyses sur des options:
    - Matrices de P&L
    - Tests de stress
    
    """
    
    def __init__(self, option, pricer, pricing_function: Callable = None, params=None):
        """
        Args:
            option: l'objet option
            pricer: l'objet pricer
            pricing_function: fonction à utiliser pour pricer l'option en fonction d'un modèle (sinon .price() par défaut)
            params: éventuels paramètres (ex: HestonParameters)
        """
        self.option = option
        self.pricer = pricer
        self.pricing_function = pricing_function if pricing_function else self.default_price
        self.params = params
        self._initial_option = copy.deepcopy(option)
        self._initial_params = copy.deepcopy(params) if params else None

    def default_price(self):
        return self.pricer.price()

    def _restore_initial_state(self):
        """Restaure l'option et les paramètres à leur état initial"""
        for attr, value in vars(self._initial_option).items():
            setattr(self.option, attr, value)

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
            x_param_name: Nom du paramètre pour l'axe X
            x_param_values: Liste des valeurs à tester pour ce paramètre
            y_param_name: Nom du paramètre pour l'axe Y 
            y_param_values: Liste des valeurs à tester pour le paramètre Y
            
        Returns:
            DataFrame contenant la matrice de P&L
        """
        initial_price = self.pricing_function()

        
        # Si on varie un seul paramètre
        if y_param_name is None or y_param_values is None:
            results = []
            for x_value in x_param_values:
                self._set_param_value(x_param_name, x_value)
                
                new_price = self.pricing_function()
                pnl = new_price - initial_price
                
                results.append({
                    x_param_name: x_value,
                    'Price': new_price,
                    'P&L': pnl,
                    'P&L (%)': (pnl / initial_price * 100) if initial_price != 0 else float('inf')
                })
            
            df = pd.DataFrame(results)
            
        else:
            # Si on varie deux paramètres (matrice)
            matrix = np.zeros((len(x_param_values), len(y_param_values)))
            
            for i, x_value in enumerate(x_param_values):
                self._set_param_value(x_param_name, x_value)
                
                for j, y_value in enumerate(y_param_values):
                    self._set_param_value(y_param_name, y_value)
                    
                    new_price = self.pricer.price()
                    pnl = new_price - initial_price
                    matrix[i, j] = pnl
            
            df = pd.DataFrame(matrix, 
                             index=x_param_values, 
                             columns=y_param_values)
            df.index.name = x_param_name
            df.columns.name = y_param_name
        
        self._restore_initial_state()
        return df
    
    def _set_param_value(self, param_name, value):
        """
        Méthode utilitaire pour définir la valeur d'un paramètre
        """
        if hasattr(self.option, param_name):
            setattr(self.option, param_name, value)
        elif self.params and hasattr(self.params, param_name):
            setattr(self.params, param_name, value)
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
        
        # Si un tableau 1D - faire un simple graphique
        if isinstance(pnl_matrix.index, pd.RangeIndex) or len(pnl_matrix.columns) <= 4:
            x_col = pnl_matrix.columns[0] 
            y_col = 'P&L' if 'P&L' in pnl_matrix.columns else 'Price' 
            
            plt.plot(pnl_matrix[x_col], pnl_matrix[y_col], 'b-', linewidth=2, marker='o')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(title)
            plt.grid(True)
            
        else:
            # Si une matrice 2D - faire une heatmap
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
        base_price = self.pricing_function()
        
        for i, scenario in enumerate(scenario_params):
            for param_name, param_value in scenario.items():
                self._set_param_value(param_name, param_value)
            
            try:
                start_time = time.time()
                stress_price = self.pricing_function()
                calc_time = time.time() - start_time
                
                # Calculer le P&L absolu et relatif
                pnl_abs = stress_price - base_price
                pnl_rel = (pnl_abs / base_price * 100) if base_price != 0 else float('inf')
                
                result = {
                    'Scenario': scenario_names[i] if scenario_names and i < len(scenario_names) else f"Scenario {i+1}",
                    'Price': stress_price,
                    'P&L': pnl_abs,
                    'P&L (%)': pnl_rel,
                    'Calc Time (s)': calc_time
                }
                
                for param_name, param_value in scenario.items():
                    result[param_name] = param_value
                
                results.append(result)
            
            except Exception as e:
                print(f"Erreur dans le scénario {i+1}: {str(e)}")
                
            self._restore_initial_state()
        
        df = pd.DataFrame(results)
        
        main_cols = ['Scenario', 'Price', 'P&L', 'P&L (%)', 'Calc Time (s)']
        other_cols = [col for col in df.columns if col not in main_cols]
        df = df[
            [col for col in main_cols if col in df.columns] + 
            [col for col in other_cols if col in df.columns]
        ]
        
        return df