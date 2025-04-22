# hestonpricer/utils/stats.py
import numpy as np
import math
from scipy.stats import norm

class Stats:
    @staticmethod
    def mean(data):
        """Calculate the mean of an array"""
        return np.mean(data)
    
    @staticmethod
    def variance(data):
        """Calculate the variance of an array"""
        return np.var(data, ddof=1)  # ddof=1 for sample variance
    
    @staticmethod
    def standard_deviation(data):
        """Calculate the standard deviation of an array"""
        return np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    
    @staticmethod
    def normal_cdf(x):
        """Calculate the standard normal cumulative distribution function"""
        '''
        if x < 0:
            return 1 - Stats.normal_cdf(-x)
        
        # Using the approximation formula # Approximation vue en cours de c# pour pas utiliser les librairies
        #  mais sera + précis avec les librairies de Python, à changer ? 
        b1 = 0.31938153
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        k = 1 / (1 + 0.2316419 * x)
        pdf = np.exp(-x * x / 2.0) / np.sqrt(2 * np.pi)
        return 1.0 - pdf * (b1 * k + b2 * k**2 + b3 * k**3 + b4 * k**4 + b5 * k**5)'''
        return norm.cdf(x)
    
    @staticmethod
    def correlation(a, b):
        """Calculate the correlation between two arrays"""
        if len(a) != len(b):
            raise ValueError("Arrays must be of the same length.")
        
        return np.corrcoef(a, b)[0, 1]
