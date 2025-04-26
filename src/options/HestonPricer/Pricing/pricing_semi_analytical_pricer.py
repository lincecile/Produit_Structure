# hestonpricer/pricing/semi_analytical_pricer.py
import numpy as np
import cmath
from ..Pricing.pricing_pricer_base import PricerBase

class SemiAnalyticalPricer(PricerBase):
    def __init__(self, option, heston_parameters):
        """
        Semi-analytical pricer for Heston model
        
        Args:
            option: Option object
            heston_parameters: Heston model parameters
        """
        super().__init__(option, heston_parameters=heston_parameters)
    
    def _characteristic_function(self, phi):
        """
        Calculate characteristic function for Heston model
        
        Args:
            phi: Complex number
            
        Returns:
            complex: Characteristic function value
        """
        b = self.heston_parameters.kappa + self.heston_parameters.lmbda
        kappa = self.heston_parameters.kappa
        theta = self.heston_parameters.theta
        sigma = self.heston_parameters.sigma
        rho = self.heston_parameters.rho
        V0 = self.heston_parameters.v0
        r = self.option.risk_free_rate
        T = self.option.maturity
        S0 = self.option.spot_price
        K = self.option.strike
        
        i = complex(0, 1.0)
        iRhoSigmaPhi = i * rho * sigma * phi
        
        d = cmath.sqrt((iRhoSigmaPhi - b) ** 2 + sigma ** 2 * (phi ** 2 + i * phi))
        g = (b - iRhoSigmaPhi + d) / (b - iRhoSigmaPhi - d)
        
        first_term = cmath.exp(r * phi * i * T) * S0 ** (i * phi) * ((1 - g * cmath.exp(d * T)) / (1 - g)) ** (-2 * theta * kappa / sigma / sigma)
        exp_term = cmath.exp(theta * kappa * T / sigma / sigma * (b - iRhoSigmaPhi + d) + V0 / sigma / sigma * (b - iRhoSigmaPhi + d) * ((1 - cmath.exp(d * T)) / (1 - g * cmath.exp(d * T))))
        
        return first_term * exp_term
    
    def _integrand(self, phi):
        """
        Calculate integrand for option price calculation
        
        Args:
            phi: Real number
            
        Returns:
            complex: Integrand value
        """
        r = self.option.risk_free_rate
        T = self.option.maturity
        K = self.option.strike
        
        i = complex(0, 1.0)
        first_term = np.exp(r * T) * self._characteristic_function(phi - i)
        second_term = -K * self._characteristic_function(phi)
        denominator = i * phi * K ** (phi * i)
        
        return (first_term + second_term) / denominator
    
    def price(self):
        """
        Calculate option price using semi-analytical method
        
        Returns:
            float: Option price
        """
        r = self.option.risk_free_rate
        T = self.option.maturity
        K = self.option.strike
        S0 = self.option.spot_price
        
        max_phi = 1000
        N = 100000
        integral = complex(0, 0)
        d_phi = max_phi / N
        
        for i in range(N):
            phi = d_phi * (2 * i + 1) / 2
            integral += self._integrand(phi) * d_phi
        
        return (S0 - K * np.exp(-r * T)) / 2.0 + 1.0 / np.pi * integral.real