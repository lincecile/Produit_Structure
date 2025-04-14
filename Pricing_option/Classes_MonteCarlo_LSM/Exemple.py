import numpy as np
import pandas as pd
from module_regression import RegressionEstimator

def compute_cash_flows(stock_price_paths, K=1.1, r=0.06):
    np.random.seed(42)
    T = stock_price_paths.shape[1] - 1  # Nombre de périodes
    
    # Initialisation des cash flows
    CF_matrix = np.zeros_like(stock_price_paths)
    CF_matrix[:, -1] = np.maximum(0, K - stock_price_paths[:, -1])  # Payoff final
    
    discount_factor = np.exp(-r)  # Facteur d'actualisation constant
    
    for t in range(T - 1, 0, -1):
        intrinsic_value = np.maximum(0, K - stock_price_paths[:, t])
        in_the_money = intrinsic_value > 0
        
        if np.any(in_the_money):  # Vérifie s'il y a des valeurs ITM
            X = stock_price_paths[in_the_money, t].reshape(-1, 1)
            Y = CF_matrix[in_the_money, t + 1] * discount_factor
            estimator = RegressionEstimator(X, Y, degree=2)
            continuation_value = np.zeros_like(intrinsic_value)
            continuation_value[in_the_money] = estimator.get_estimator(X)
        else:
            continuation_value = np.zeros_like(intrinsic_value)

        exercise = intrinsic_value > continuation_value
        CF_matrix[:, t] = np.where(exercise, intrinsic_value, 0)
        CF_matrix[:, t + 1] *= ~exercise  # Annule les cash flows futurs si exercé
    
    # Trouver la position du dernier cash flow non nul pour chaque trajectoire
    last_nonzero_indices = (CF_matrix != 0).cumsum(axis=1).argmax(axis=1)

    # Extraire le dernier cash flow non nul
    last_CF = CF_matrix[np.arange(CF_matrix.shape[0]), last_nonzero_indices]

    # Actualiser en fonction du temps
    discounted_CF = last_CF * np.exp(-r * last_nonzero_indices)

    return np.mean(discounted_CF)

# Exemple d'utilisation
test_stock_price_paths = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34]
])

discounted_CF= compute_cash_flows(test_stock_price_paths)
print(discounted_CF)
