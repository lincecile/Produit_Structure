#%% Imports
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple

from src.options.Pricing_option.Classes_Both.module_barriere import TypeBarriere, Barriere, DirectionBarriere
from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche
from src.options.Pricing_option.Classes_Both.module_option import Option
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_brownian import Brownian
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_regression import RegressionEstimator
import numpy as np
import pandas as pd

from numpy.polynomial import Polynomial, Laguerre, Hermite

#%% Classes

class LSM_method : 
    """Classe utilisée pour calculer le prix d'une option."""
    
    def __init__(self, option: Option):
        self.option = option
    
    def underlying_paths(self, S0: float, taux_interet: np.ndarray, sigma: float, W: np.ndarray, timedelta: np.ndarray) -> np.ndarray:
        """Génère les trajectoires de prix du sous-jacent avec la première colonne initialisée à S0."""
        S = S0 * np.exp((taux_interet - sigma**2/2) * timedelta + sigma * W)
        S[:, 0] = S0
        return S
    
    def __calcul_position_div(self, market: DonneeMarche, brownian: Brownian) -> int:
        """Calcule la position du dividende dans l'arbre."""
        nb_jour_detachement = (market.dividende_ex_date - self.option.date_pricing).days
        return nb_jour_detachement / 365 / brownian.step
    
    def adjust_for_dividends(self, S_T: np.ndarray, market: DonneeMarche, brownian: Brownian, W: np.ndarray, timedelta: np.ndarray) -> None:
        """ Ajuste les trajectoires pour prendre en compte les dividendes. """
        position_div = self.__calcul_position_div(market=market, brownian=brownian)
        position_div = int(position_div) #if np.abs(position_div - int(position_div)) < 0.01 else int(position_div) + 1 
        if position_div < brownian.nb_step:
            S_T[:, position_div] -= market.dividende_montant
            S_T[:, position_div + 1:] = S_T[:, position_div][:, np.newaxis] * np.exp(
                (market.taux_interet - market.volatilite**2 / 2) * 
                (timedelta[position_div + 1:] - timedelta[position_div]) +
                market.volatilite * (W[:, position_div + 1:] - W[:, position_div][:, np.newaxis]))

    def antithetic_mode(self, S0: float, taux_interet: np.ndarray, sigma: float, W: np.ndarray, 
                        timedelta: np.ndarray, market: DonneeMarche, brownian: Brownian) -> np.ndarray:
        """ Applique la méthode antithétique. """
        W_neg = -W
        S_T_pos = self.underlying_paths(S0, taux_interet, sigma, W, timedelta)
        S_T_neg = self.underlying_paths(S0, taux_interet, sigma, W_neg, timedelta)
        
        if market.dividende_montant > 0:
            self.adjust_for_dividends(S_T_pos, market, brownian, W, timedelta)
            self.adjust_for_dividends(S_T_neg, market, brownian, W_neg, timedelta)
        
        S_T_antithetic = np.concatenate([S_T_pos, S_T_neg], axis=0)
        
        return S_T_antithetic
    
    def vector_method(self, S0: float, taux_interet: np.ndarray, sigma: float, q: float, 
                      market: DonneeMarche, brownian: Brownian, antithetic: bool) -> np.ndarray:
        """ Calcule les trajectoires avec la méthode vectorielle. """
        W = brownian.Vecteur()
        timedelta = np.array([brownian.step * i for i in range(brownian.nb_step+1)])
        
        if antithetic:
            return self.antithetic_mode(S0, taux_interet, sigma, W, timedelta, market, brownian)
        
        S_T = self.underlying_paths(S0, taux_interet, sigma, W, timedelta)
        if q > 0:
            self.adjust_for_dividends(S_T, market, brownian, W, timedelta)

        return S_T
    
    def scalar_method(self, S0: float, taux_interet: np.ndarray, sigma: float, q: float, 
                      T: float, market: DonneeMarche, brownian: Brownian) -> np.ndarray:
        """ Calcule les trajectoires avec la méthode scalaire. """
        S_T = np.ones((brownian.nb_trajectoire, brownian.nb_step+1)) * S0
        position_div = self.__calcul_position_div(market, brownian)
        for i in range(brownian.nb_trajectoire):
            W = brownian.Scalaire()
            for j in range(1, brownian.nb_step+1):
                S_T[i, j] = S_T[i, j-1] * np.exp((taux_interet[j-1] - sigma**2 / 2) * brownian.step + sigma * (W[j] - W[j-1]))
                if j == int(position_div):
                    S_T[i, j] -= market.dividende_montant
        return S_T
    
    def Price(self, market: DonneeMarche, brownian: Brownian, method: str = 'vector', antithetic: bool = False) -> np.ndarray:
        """
        Calcule le val_intriseque de l'option en utilisant un mouvement brownien.
        """
        S0 = market.prix_spot
        taux_interet = market.taux_interet
        sigma = market.volatilite
        q = market.dividende_montant  
        T = self.option.maturity

        if method == 'vector':
            return self.vector_method(S0, taux_interet, sigma, q, market, brownian, antithetic)
        else:
            return self.scalar_method(S0, taux_interet, sigma, q, T, market, brownian)

    def check_barrier_condition(self, Spot_simule: np.ndarray) -> bool:
        """
        Vérifie si la condition de barrière est satisfaite pour un chemin donné,
        en utilisant les classes TypeBarriere et DirectionBarriere.
        
        Args:
            path: Trajectoire de prix du sous-jacent
            
        Returns:
            bool: True si la condition de barrière est satisfaite, False sinon
        """
        
        if self.option.barriere is None or self.option.barriere.direction_barriere is not None:
            return np.ones(Spot_simule.shape[0], dtype=bool)
        
        barrier_value = self.option.barriere.niveau_barriere

        if self.option.barriere.type_barriere is TypeBarriere.knock_in:
            if self.option.barriere.direction_barriere is DirectionBarriere.up:
                return np.any(Spot_simule >= barrier_value, axis=1)
            elif self.option.barriere.direction_barriere is DirectionBarriere.down:
                return np.any(Spot_simule <= barrier_value, axis=1)

        elif self.option.barriere.type_barriere is TypeBarriere.knock_out:
            if self.option.barriere.direction_barriere is DirectionBarriere.up:
                return ~np.any(Spot_simule >= barrier_value, axis=1)
            elif self.option.barriere.direction_barriere is DirectionBarriere.down:
                return ~np.any(Spot_simule <= barrier_value, axis=1)

        return np.ones(Spot_simule.shape[0], dtype=bool)


    def check_barrier_condition_up_to_t(self, Spot_simule: np.ndarray, t: int) -> np.ndarray:
        """
        Vérifie vectoriellement la condition de barrière jusqu'à un instant t inclus.

        Args:
            Spot_simule: np.ndarray (nb_trajectoires, nb_steps+1)
            t: int, temps jusqu'auquel vérifier la condition (inclus)

        Returns:
            np.ndarray: Tableau booléen indiquant si chaque trajectoire respecte la condition jusqu'à t
        """
        if self.option.barriere is None or self.option.barriere.direction_barriere is None:
            return np.ones(Spot_simule.shape[0], dtype=bool)

        barrier_value = self.option.barriere.niveau_barriere
        Spot_sub = Spot_simule[:, :t+1]  # on découpe seulement jusqu'à t

        if self.option.barriere.type_barriere is TypeBarriere.knock_in:
            if self.option.barriere.direction_barriere is DirectionBarriere.up:
                return np.any(Spot_sub >= barrier_value, axis=1)
            elif self.option.barriere.direction_barriere is DirectionBarriere.down:
                return np.any(Spot_sub <= barrier_value, axis=1)

        elif self.option.barriere.type_barriere is TypeBarriere.knock_out:
            if self.option.barriere.direction_barriere is DirectionBarriere.up:
                return ~np.any(Spot_sub >= barrier_value, axis=1)
            elif self.option.barriere.direction_barriere is DirectionBarriere.down:
                return ~np.any(Spot_sub <= barrier_value, axis=1)

        return np.ones(Spot_simule.shape[0], dtype=bool)
    
    def compute_intrinsic_value(self, Spot_simule: np.ndarray) -> np.ndarray:
        
        barrier_conditions = self.check_barrier_condition(Spot_simule)
        
        if self.option.call:
            return np.maximum(Spot_simule[:, -1] - self.option.prix_exercice, 0.0) * barrier_conditions
        return np.maximum(self.option.prix_exercice - Spot_simule[:, -1], 0.0) * barrier_conditions
    
    def lsm_algorithm(self, CF_Vect: np.ndarray, Spot_simule: np.ndarray, brownian: Brownian, 
                      market: DonneeMarche, poly_degree: int, model_type: str) -> np.ndarray:
        
        for t in range(brownian.nb_step - 1, 0, -1): 
            
            CF_next_actualise = CF_Vect * np.exp(-market.taux_interet[t] * self.option.maturity / brownian.nb_step) # CF au temps suivant actualisé
            
            # Calcul des valeurs intrinsèques à l'instant t avec vérification des barrières
            barrier_conditions = self.check_barrier_condition_up_to_t(Spot_simule, t)

            if self.option.barriere is not None or self.option.barriere.direction_barriere is not None:
                for i in range(Spot_simule.shape[0]):
                    # Vérifier la condition jusqu'au temps t inclus
                    barrier_conditions[i] = self.check_barrier_condition(Spot_simule[i, :t+1])
        
            val_intriseque = self.compute_intrinsic_value(Spot_simule[:, t].reshape(-1, 1))                    
            in_the_money = val_intriseque > 0                                                   # Chemins dans la monnaie en t    
            CF_Vect = CF_next_actualise.copy()                                                 # CF en t1 actualisé en t par défaut
            
            # Si des chemins sont dans la monnaie en t, on fait la regression
            if np.any(in_the_money):  
                
                X = Spot_simule[in_the_money, t]       # prix du sous jacent en t
                Y = CF_next_actualise[in_the_money]   # CF des chemins dans la monnaie en t1 actualisé en t
                
                # CF espérés en t pour les chemins dans la monnaie si on n'exerce pas
                continuation_values = RegressionEstimator(X,Y, degree=poly_degree, model_type=model_type).Regression()

                # Exercice anticipé en t si valeur en t est supérieure à la valeur espérée
                exercise = val_intriseque[in_the_money] > continuation_values
                
                # Mise à jour des CF en t pour les chemins dans la monnaie
                CF_Vect[in_the_money]  = np.where(exercise, val_intriseque[in_the_money], CF_next_actualise[in_the_money])
        
        # Valeur en t0
        CF_Vect = CF_Vect * np.exp(-market.taux_interet[0] * self.option.maturity / brownian.nb_step)
        
        return CF_Vect
    
    def LSM(self, brownian: Brownian, market: DonneeMarche, poly_degree: int = 2, 
            model_type: str = "Polynomial", method: str = 'vector', 
            antithetic: bool = False, print_info: bool = False) -> Tuple[float, float, Tuple[float, float]]:
        

        if isinstance(market.taux_interet, (float, int)):
            # Convertit en array avec la même valeur répétée pour chaque pas de temps
            market.taux_interet = np.full(brownian.nb_step+1 , market.taux_interet)
        elif isinstance(market.taux_interet, np.ndarray) and len(market.taux_interet) != brownian.nb_step + 1:
            # Rebuild si c'est un array mais pas de la bonne taille, utile pour le graphique
            market.taux_interet = np.full(brownian.nb_step + 1, market.taux_interet[0])

        # Prix du sous-jacent simulé
        Spot_simule = self.Price(market, brownian, method=method, antithetic=antithetic)

        # Calcul valeur intrinsèque à chaque pas de temps
        val_intriseque = self.compute_intrinsic_value(Spot_simule)

        antithetic_info = 'antithetic' if antithetic else 'non antithetic'
        euro_americain_info = 'américaine' if self.option.americaine else 'européenne'

        # Valeur de l'option européenne
        if not self.option.americaine:
            val_intriseque = val_intriseque * np.exp(-market.taux_interet[-1] * self.option.maturity)
            prix, std_prix, intervalle = self.calculate_price_statistics(val_intriseque, len(val_intriseque), antithetic_info, euro_americain_info, method, print_info=print_info)
            return (prix, std_prix, intervalle)
        
        # Vecteur des cash flows
        CF_Vect = val_intriseque.copy()
        CF_Vect = self.lsm_algorithm(CF_Vect, Spot_simule, brownian, market, poly_degree, model_type)

        if not antithetic:
            prix, std_prix, intervalle = self.calculate_price_statistics(CF_Vect, len(CF_Vect), antithetic_info, euro_americain_info, method, print_info=print_info)
            return (prix, std_prix, intervalle)
        
        moitie = len(CF_Vect) // 2
        CF_vect_final = (CF_Vect[:moitie] + CF_Vect[moitie:]) / 2
        prix, std_prix, intervalle = self.calculate_price_statistics(CF_vect_final, len(CF_vect_final), antithetic_info, euro_americain_info, method, print_info=print_info)
        
        return (prix, std_prix, intervalle)

    def calculate_price_statistics(self, CF_values: np.ndarray, nb_chemins: int, 
                                  mode: str, type_option: str, method: str, 
                                  print_info: bool = False) -> Tuple[float, float, Tuple[float, float]]:
        prix = np.mean(CF_values)
        std_prix = np.std(CF_values) / np.sqrt(len(CF_values))
        intervalle = (prix - 2 * std_prix, prix + 2 * std_prix)
        if print_info:
            print(f"Nb chemins {mode}: {nb_chemins}")
            print(f"Prix min {mode, type_option, method}: {prix - 2 * std_prix}")
            print(f"Prix max {mode, type_option, method}: {prix + 2 * std_prix}")
        return prix, std_prix, intervalle
        
# %%
