# =====================
# module_option.py
# =====================
from dataclasses import dataclass
import datetime as dt
from typing import Optional
import numpy as np

@dataclass
class Option:
    call: bool
    prix_exercice: float
    maturity: float
    americaine: bool = False
    barriere: Optional[float] = None
    type_barriere: Optional[str] = None  # "up-in", "up-out", "down-in", "down-out"
    date_pricing: Optional[dt.datetime] = None

    def barrier_condition(self, path: np.ndarray) -> bool:
        if self.barriere is None or self.type_barriere is None:
            return True  # pas de barrière, toujours valide

        if self.type_barriere == "up-in":
            return np.any(path >= self.barriere)
        elif self.type_barriere == "down-in":
            return np.any(path <= self.barriere)
        elif self.type_barriere == "up-out":
            return not np.any(path >= self.barriere)
        elif self.type_barriere == "down-out":
            return not np.any(path <= self.barriere)
        else:
            raise ValueError("Type de barrière non reconnu")

    def payoff(self, path: np.ndarray) -> float:
        if not self.barrier_condition(path):
            return 0.0
        spot = path[-1]
        if self.call:
            return max(spot - self.prix_exercice, 0.0)
        return max(self.prix_exercice - spot, 0.0)

    def payoff_array(self, paths: np.ndarray) -> np.ndarray:
        payoffs = np.zeros(paths.shape[0])
        for i, path in enumerate(paths):
            payoffs[i] = self.payoff(path)
        return payoffs
