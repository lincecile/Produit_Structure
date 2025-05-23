#%% Imports

from enum import Enum

#%% Classes enum

class ConventionBaseCalendaire(Enum) :
    _365 = 365
    _360 = 360
    _252 = 252
    _257 = 257
    _366 = 366

class TypeBarriere(Enum) : 
    knock_in = "Knock-in"
    knock_out = "Knock-out"
    
class DirectionBarriere(Enum) : 
    up = "Up"
    down = "Down"

class SensOption(Enum) : 
    short = "Short"
    long = "Long"

class StratOption(Enum) : 
    callspread = "Call Spread"
    putspread = "Put Spread"
    strangle = "Strangle"
    straddle = "Straddle"
    butterfly = "Butterfly"
    collar = "Collar"
    forward = "Forward"

class StratStructured(Enum) : 
    capitalprotectednote = "Capital protected note"
    reverseconvertible = "Reverse convertible"
    barrierdigit = "Barrier digital"
    autocallathena = "Autocall Athena"

class MethodeCalcul(Enum) : 
    vector = "Vectorielle"
    scalar = "Scalaire"

class RegType(Enum) : 
    polynomial = "Polynomial"
    laguerre = "Laguerre"
    hermite = "Hermite"
    legendre = "Legendre"
    chebyshev = "Chebyshev"
    linear = "Linear"
    logarithmic = "Logarithmic"
    exponential = "Exponential"
            

class ModelMetrics(Enum):
    bs = "Black-Scholes"
    arbre = "Arbre Trinomial"
    lsm = "LSM"
    heston = "Heston"