#%% Imports
import datetime as dt
from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..Classes_TrinomialTree.module_barriere import Barriere

from ..Classes_MonteCarlo_LSM.module_brownian import Brownian
from ..Classes_MonteCarlo_LSM.module_regression import RegressionEstimator
#%% Classes

@dataclass
class Option : 
    """Classe utilisée pour représenter une option et ses paramètres.
    """
    
    maturite : dt.date
    prix_exercice : float
    barriere : Optional[Barriere] = None
    americaine : bool = False
    call : bool = True
    date_pricing : dt.date = dt.date.today() 
    type_barriere : Optional[str] = None  # "up-in", "up-out", "down-in", "down-out"


    @property
    def maturity(self) -> float:
        return (self.maturite - self.date_pricing).days / 365.0 
    
    @maturity.setter
    def maturity(self, value: float):
        self.maturite = self.date_pricing + dt.timedelta(days=int(value * 365))

    def barrier_condition(self, path: np.ndarray) -> bool:
        """
        Vérifie si la condition de barrière est satisfaite pour un chemin donné.
        
        Args:
            path: Trajectoire de prix du sous-jacent
            
        Returns:
            bool: True si la condition de barrière est satisfaite, False sinon
        """
        if self.barriere is None or self.type_barriere is None:
            return True  # pas de barrière, toujours valide

        barrier_value = self.barriere.niveau if isinstance(self.barriere, Barriere) else self.barriere

        if self.type_barriere == "up-in":
            return np.any(path >= barrier_value)
        elif self.type_barriere == "down-in":
            return np.any(path <= barrier_value)
        elif self.type_barriere == "up-out":
            return not np.any(path >= barrier_value)
        elif self.type_barriere == "down-out":
            return not np.any(path <= barrier_value)
        else:
            raise ValueError(f"Type de barrière non reconnu: {self.type_barriere}")
            
    def payoff(self, spot: float) -> float:
        """
        Calcule le payoff de l'option pour un prix spot donné.
        
        Args:
            spot: Prix spot du sous-jacent
            
        Returns:
            float: Payoff de l'option
        """
        if self.call:
            return max(spot - self.prix_exercice, 0.0)
        return max(self.prix_exercice - spot, 0.0)