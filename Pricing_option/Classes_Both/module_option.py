#%% Imports
import datetime as dt
from dataclasses import dataclass
from typing import Optional

from Classes_TrinomialTree.module_barriere import Barriere

from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_regression import RegressionEstimator
import numpy as np
import pandas as pd
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

    @property
    def maturity(self) -> float:
        return (self.maturite - self.date_pricing).days / 365.0 
    
    @maturity.setter
    def maturity(self, value: float):
        self.maturite = self.date_pricing + dt.timedelta(days=int(value * 365))