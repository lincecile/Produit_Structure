# hestonpricer/utils/random_number_generator.py
import numpy as np

class RandomNumberGenerator:
    @staticmethod
    def generate_correlated_normals(rho, random_state=None):
        """
        Generate two correlated normal random variables
        
        Args:
            rho: Correlation coefficient (-1 <= rho <= 1)
            random_state: Random state (optional)
            
        Returns:
            tuple: Two correlated normal random variables
        """
        if rho < -1 or rho > 1:
            raise ValueError("Correlation coefficient must be between -1 and 1.")
        
        # Set random state if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate two independent standard normal random variables
        u1 = np.random.random()
        u2 = np.random.random()
        
        # Box-Muller transform to get standard normal random variables
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)
        
        # Create correlation
        x = z0
        y = rho * z0 + np.sqrt(1 - rho * rho) * z1
        
        return np.array([x, y])
