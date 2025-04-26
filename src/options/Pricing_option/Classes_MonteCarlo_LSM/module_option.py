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
    date_pricing: Optional[dt.datetime] = None
