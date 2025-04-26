# =====================
# module_regression.py
# =====================

from numpy.polynomial import Polynomial, Laguerre, Hermite, Legendre, Chebyshev
import numpy as np
from typing import Dict, Callable
from sklearn.linear_model import LinearRegression
class RegressionEstimator:
    """
    Classe permettant d'estimer une régression selon différents modèles polynomiaux,
    utilisée pour estimer la valeur de continuation dans le pricing d'options américaines
    et d'options à barrière.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, degree: int = 2, model_type: str = "Polynomial"):
        self.X = X
        self.Y = Y
        self.degree = degree
        self.model_type = model_type

    def _polynomial_regression(self) -> np.ndarray:
        coeffs = np.polynomial.polynomial.polyfit(self.X, self.Y, self.degree)
        return Polynomial(coeffs)(self.X)

    def _laguerre_regression(self) -> np.ndarray:
        coeffs = np.polynomial.laguerre.lagfit(self.X, self.Y, self.degree)
        return Laguerre(coeffs)(self.X)

    def _hermite_regression(self) -> np.ndarray:
        coeffs = np.polynomial.hermite.hermfit(self.X, self.Y, self.degree)
        return Hermite(coeffs)(self.X)

    def _legendre_regression(self) -> np.ndarray:
        coeffs = np.polynomial.legendre.legfit(self.X, self.Y, self.degree)
        return Legendre(coeffs)(self.X)

    def _chebyshev_regression(self) -> np.ndarray:
        coeffs = np.polynomial.chebyshev.chebfit(self.X, self.Y, self.degree)
        return Chebyshev(coeffs)(self.X)

    def _linear_regression(self) -> np.ndarray:
        coeffs = np.polyfit(self.X, self.Y, 1)
        return np.polyval(coeffs, self.X)

    def _logarithmic_regression(self) -> np.ndarray:
        log_X = np.log(self.X)
        coeffs = np.polyfit(log_X, self.Y, 1)
        return np.polyval(coeffs, log_X)

    def _exponential_regression(self) -> np.ndarray:
        log_Y = np.log(self.Y)
        coeffs = np.polyfit(self.X, log_Y, 1)
        return np.exp(np.polyval(coeffs, self.X))

    def Regression(self) -> np.ndarray:
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
