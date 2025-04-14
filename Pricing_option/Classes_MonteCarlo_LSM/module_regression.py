from sklearn.linear_model import LinearRegression
from numpy.polynomial import Polynomial, Laguerre, Hermite, Legendre, Chebyshev
import numpy as np
from typing import Union, Literal, Optional, List, Tuple, Any, Dict, Callable, Type

class RegressionEstimator:
    """
    Classe permettant d'estimer une régression selon différents modèles polynomiaux.
    Utilisée dans l'algorithme LSM pour estimer la valeur de continuation des options américaines.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, degree: int = 2, 
                 model_type = "Polynomial"):
        """
        Initialise un estimateur de régression.
        
        Args:
            X: Variable explicative (généralement le prix du sous-jacent)
            Y: Variable à expliquer (généralement les cash-flows futurs actualisés)
            degree: Degré du polynôme pour la régression
            model_type: Type de modèle de régression à utiliser
        """
        self.X = X
        self.Y = Y
        self.degree = degree
        self.model_type = model_type

    def _polynomial_regression(self) -> np.ndarray:
        """
        Effectue une régression polynomiale standard.
        
        Returns:
            Valeurs prédites pour les X d'entrée
        """
        coeffs: np.ndarray = np.polynomial.polynomial.polyfit(self.X, self.Y, self.degree)
        linreg: Polynomial = Polynomial(coeffs)
        return linreg(self.X)
    
    def _laguerre_regression(self) -> np.ndarray:
        """
        Effectue une régression sur base de polynômes de Laguerre.
        
        Returns:
            Valeurs prédites pour les X d'entrée
        """
        coeffs: np.ndarray = np.polynomial.laguerre.lagfit(self.X, self.Y, self.degree)
        linreg: Laguerre = Laguerre(coeffs)
        return linreg(self.X)
    
    def _hermite_regression(self) -> np.ndarray:
        """
        Effectue une régression sur base de polynômes d'Hermite.
        
        Returns:
            Valeurs prédites pour les X d'entrée
        """
        coeffs: np.ndarray = np.polynomial.hermite.hermfit(self.X, self.Y, self.degree)
        linreg: Hermite = Hermite(coeffs)
        return linreg(self.X)
    
    def _legendre_regression(self) -> np.ndarray:
        """
        Effectue une régression sur base de polynômes de Legendre.
        
        Returns:
            Valeurs prédites pour les X d'entrée
        """
        coeffs: np.ndarray = np.polynomial.legendre.legfit(self.X, self.Y, self.degree)
        linreg: Legendre = Legendre(coeffs)
        return linreg(self.X)
    
    def _chebyshev_regression(self) -> np.ndarray:
        """
        Effectue une régression sur base de polynômes de Chebyshev.
        
        Returns:
            Valeurs prédites pour les X d'entrée
        """
        coeffs: np.ndarray = np.polynomial.chebyshev.chebfit(self.X, self.Y, self.degree)
        linreg: Chebyshev = Chebyshev(coeffs)
        return linreg(self.X)
    
    def _linear_regression(self) -> np.ndarray:
        """
        Effectue une régression linéaire simple.
        
        Returns:
            Valeurs prédites pour les X d'entrée
        """
        coeffs: np.ndarray = np.polyfit(self.X, self.Y, 1)
        return np.polyval(coeffs, self.X)
    
    def _logarithmic_regression(self) -> np.ndarray:
        """
        Effectue une régression logarithmique (log(X) vs Y).
        
        Returns:
            Valeurs prédites pour les X d'entrée
        """
        log_X: np.ndarray = np.log(self.X)
        coeffs: np.ndarray = np.polyfit(log_X, self.Y, 1)
        return np.polyval(coeffs, log_X)
    
    def _exponential_regression(self) -> np.ndarray:
        """
        Effectue une régression exponentielle (X vs log(Y)).
        
        Returns:
            Valeurs prédites pour les X d'entrée
        """
        log_Y: np.ndarray = np.log(self.Y)
        coeffs: np.ndarray = np.polyfit(self.X, log_Y, 1)
        return np.exp(np.polyval(coeffs, self.X))

    def Regression(self) -> np.ndarray:
        """
        Sélectionne et applique la méthode de régression appropriée.
        
        Returns:
            Valeurs prédites pour les X d'entrée selon le modèle choisi
        """
        regression_methods: Dict[str, Callable[[], np.ndarray]] = {
            "Polynomial": self._polynomial_regression,
            "Laguerre": self._laguerre_regression,
            "Hermite": self._hermite_regression,
            "Legendre": self._legendre_regression,
            "Chebyshev": self._chebyshev_regression,
            "Linear": self._linear_regression,
            "Logarithmic": self._logarithmic_regression,
            "Exponential": self._exponential_regression
        }
        
        method = regression_methods.get(self.model_type)
        if method:
            return method()
        else:
            raise ValueError(f"Modèle '{self.model_type}' non reconnu.")