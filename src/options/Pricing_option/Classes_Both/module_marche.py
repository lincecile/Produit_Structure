#%% Imports

import datetime as dt
from dataclasses import dataclass
import numpy as np

#%% Classes

@dataclass
class DonneeMarche : 
    """Classe utilisée pour représenter les données de marché.
    """
    
    date_debut : dt.date
    prix_spot : float
    volatilite : float
    taux_interet : np.ndarray | float
    taux_actualisation : np.ndarray | float
    dividende_ex_date : dt.date
    dividende_montant : float = 0
    dividende_rate : float = 0

    def has_dividend(self) -> bool:
        """
        Vérifie si un dividende est défini.
        
        Returns:
            bool: True si un dividende est défini, False sinon
        """
        return self.dividende_montant > 0
    
    def get_dividend_step(self, step_size: float, date_pricing: dt.date) -> int:
        """
        Calcule le step correspondant à la date de dividende.
        
        Args:
            step_size: Taille d'un pas de temps en années
            date_pricing: Date de valorisation
            
        Returns:
            int: Step correspondant à la date de dividende
        """
        if not self.has_dividend():
            return -1
            
        days = (self.dividende_ex_date - date_pricing).days
        return int(days / 365 / step_size)