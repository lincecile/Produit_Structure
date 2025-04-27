from dataclasses import dataclass
from typing import Callable
from src.options.Pricing_option.Classes_Both.module_option import Option
from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_brownian import Brownian
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_LSM import LSM_method
from copy import deepcopy
from typing import Union
import datetime as dt

"""
Une classe de données représentant les paramètres d'une option dérivée.
Attributs :
    price (float) : Prix actuel de l'actif sous-jacent.
    strike (float) : Prix d'exercice de l'option.
    maturity (float) : Temps restant avant l'échéance (en années).
    r (float) : Taux d'intérêt sans risque.
    sigma (float) : Volatilité de l'actif sous-jacent.
"""
@dataclass
class OptionDerivativesParameters:
    option: Option  # Objet option
    market: DonneeMarche  # Objet marché
    
    # getter
    def __getitem__(self,  item: str)-> Union[float, dt.date]:
        if item == "price":
            return self.market.prix_spot
        elif item == "strike":
            return self.option.prix_exercice
        elif item == "maturite":
            return self.option.maturite
        elif item == "maturity":
            return self.option.maturity
        elif item == "r":
            return self.market.taux_interet
        elif item == "sigma":
            return self.market.volatilite
        else:
            raise KeyError(f"Paramètre inconnu: {item}")

    # setter
    def __setitem__(self, item: str, value: Union[float, dt.date]) -> None:
        if item == "price":
            self.market.prix_spot = value
        elif item == "strike":
            self.option.prix_exercice = value
        elif item == "maturite":
            self.option.maturite = value
        elif item == "maturity":
            self.option.maturity = value
        elif item == "r":
            self.market.taux_interet = value
        elif item == "sigma":
            self.market.volatilite = value
        else:
            raise KeyError(f"Paramètre inconnu: {item}")

    

class OneDimDerivative: 
    """
    Classe permettant de calculer la dérivée première d'une fonction par rapport à un paramètre donné.
    Attributs :
        f : Callable
            La fonction pour laquelle la dérivée est calculée.
        parameters : OptionDerivativesParameters
            Paramètres utilisés pour l'évaluation de la fonction.
        shift : float
            Pas utilisé dans l'approximation par différences finies.
    Méthodes :
        first(along: str) -> float
            Calcule la dérivée première par rapport au paramètre spécifié.
    """
    
    def __init__(self, function:Callable, parameters: OptionDerivativesParameters, shift:float=1, brownian=None) -> None:
        self.f = function
        self.parameters = parameters
        self.shift = shift
        self.brownian = brownian
        
    # Calcul de la dérivée première avec la formule des différences finies
    def first(self, along: str) -> float:
        params_u = deepcopy(self.parameters)
        params_u[along] += self.shift
        
        params_d = deepcopy(self.parameters)
        params_d[along] -= self.shift
        return (self.f(params_u, brownian=self.brownian)[0] - self.f(params_d, brownian=self.brownian)[0]) / (2 * self.shift)

class TwoDimDerivatives:
    """
    Classe permettant de calculer une dérivée seconde croisée d'une fonction par rapport à deux paramètres.
    Attributs :
        f : Callable
            Fonction pour laquelle on veut la dérivée seconde.
        parameters : OptionDerivativesParameters
            Paramètres de la fonction.
        shift : float
            Pas de l'approximation en différences finies.
    Méthodes :
        second(along1: str, along2: str) -> float
            Calcule la dérivée seconde croisée par rapport aux deux paramètres.
"""
    def __init__(self, function: Callable, parameters: OptionDerivativesParameters, shift: float = 1,brownian=None) -> None:
        self.f = function
        self.parameters = parameters
        self.shift = shift
        self.brownian = brownian

    # Calcul de la dérivée seconde avec la formule des différences finies 
    def second(self, along1: str, along2: str) -> float:
        params_uu = deepcopy(self.parameters)
        params_uu[along1] += self.shift
        params_uu[along2] += self.shift

        params_ud = deepcopy(self.parameters)
        params_ud[along1] += self.shift
        params_ud[along2] -= self.shift

        params_du = deepcopy(self.parameters)
        params_du[along1] -= self.shift
        params_du[along2] += self.shift

        params_dd = deepcopy(self.parameters)
        params_dd[along1] -= self.shift
        params_dd[along2] -= self.shift

        numerator = (
        self.f(params_uu, brownian=self.brownian)[0]
        - self.f(params_ud, brownian=self.brownian)[0]
        - self.f(params_du, brownian=self.brownian)[0]
        + self.f(params_dd, brownian=self.brownian)[0]
    )
        denominator = 4 * self.shift * self.shift
        return numerator / denominator


class OptionDerivatives: 
    """
    Classe permettant de calculer le prix et les "Greeks" (delta, vega, theta, gamma, vomma) d'une option par méthode numérique.
    Attributs :
        option : Option
            L'option concernée.
        parameters : OptionDerivativesParameters
            Paramètres de l'option (marché + caractéristiques).
        pricer_options : dict
            src.options supplémentaires pour le pricer LSM.
    """
    def __init__(self, option: Option, market: DonneeMarche, pricer : LSM_method, pricer_options: dict = None)-> None:
        self.option = deepcopy(option)
        self.market = deepcopy(market)
        self.pricer = pricer
        self.parameters = OptionDerivativesParameters(option, market)
        self.pricer_options = pricer_options or {}
        
    # Méthode pour calculer le prix avec les paramètres de l'option
    def price(self, params: OptionDerivativesParameters, brownian : Brownian)-> float:
    # Récupération des paramètres
        self.market.prix_spot = params["price"]
        self.market.volatilite = params["sigma"]
        self.market.taux_interet = params["r"]
        self.option.prix_exercice = params["strike"]
        self.option.maturite = params["maturite"]
        self.option.maturity = params["maturity"]

        period = (self.option.maturite - self.option.date_pricing).days / 365
        brownian = Brownian(period, brownian.nb_step, brownian.nb_trajectoire, brownian.seed)
        pricer = LSM_method(self.option)

        # Pricing avec LSM
        return pricer.LSM(brownian, self.market, **self.pricer_options)

    # Calcul du delta : sensibilité au prix du sous-jacent
    def delta(self, brownian: Brownian)-> float:
        return OneDimDerivative(self.price, self.parameters, shift=1, brownian=brownian).first("price")
    
    # Calcul du vega : sensibilité à la volatilité
    def vega(self, brownian: Brownian)-> float:
        return OneDimDerivative(self.price, self.parameters, shift=0.01, brownian=brownian).first("sigma")
    
    # Calcul du theta : sensibilité au temps (à l'échéance)
    def theta(self, brownian: Brownian)-> float:
        return -OneDimDerivative(self.price, self.parameters, shift=1/365, brownian=brownian).first("maturity")
    
    # Calcul du rho : sensibilité au taux d'intérêt
    def rho(self, brownian: Brownian)-> float:
        return OneDimDerivative(self.price, self.parameters, shift=0.01, brownian=brownian).first("r")
    
    # Calcul du gamma : dérivée seconde par rapport au prix
    def gamma(self, brownian: Brownian)-> float:
        return TwoDimDerivatives(self.price, self.parameters, brownian=brownian).second("price", "price")
    
    # Calcul du vomma : dérivée seconde par rapport à la volatilité
    def vomma(self, brownian: Brownian)-> float:
        return TwoDimDerivatives(self.price, self.parameters, shift=0.01, brownian=brownian).second("sigma", "sigma")
