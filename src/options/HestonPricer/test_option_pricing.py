# hestonpricer/test_pricing.py
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd

# Import models
from .Models.models_european_option import EuropeanOption
from .Models.models_asian_option import AsianOption
from .Models.models_heston_parameters import HestonParameters

# Import pricers
from .Pricing.pricing_black_scholes_pricer import BlackScholesPricer
from .Pricing.pricing_semi_analytical_pricer import SemiAnalyticalPricer
from .Pricing.pricing_monte_carlo_pricer import MonteCarloPricer

import sys
import os

# Remonter de 3 niveaux
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
sys.path.append(parent_dir)

# Import analysis tools
from analysis_tools import AnalysisTools

def test_european_option():
    """Test pricing d'une option européenne avec différentes méthodes"""
    print("\n===== TEST OPTION EUROPÉENNE =====")
    
    # Paramètres de l'option
    spot_price = 100.0
    strike = 100.0
    maturity = 1.0
    risk_free_rate = 0.05
    is_call = True
    
    # Création de l'option européenne
    european_option = EuropeanOption(spot_price, strike, maturity, risk_free_rate, is_call)
    
    # Paramètres du modèle de Heston
    kappa = 1.0      # Vitesse de retour à la moyenne
    theta = 0.04     # Variance à long terme
    v0 = 0.04        # Variance initiale
    sigma = 0.2      # Volatilité de la volatilité
    rho = -0.7       # Corrélation
    
    heston_params = HestonParameters(kappa, theta, v0, sigma, rho)
    
    # 1. Black-Scholes pricing
    vol = np.sqrt(v0)
    bs_pricer = BlackScholesPricer(european_option, vol)
    bs_price = bs_pricer.price()
    print(f"Prix Black-Scholes: {bs_price:.4f}")
    
    # 2. Heston semi-analytical pricing
    start_time = time()
    sa_pricer = SemiAnalyticalPricer(european_option, heston_params)
    sa_price = sa_pricer.price()
    sa_time = time() - start_time
    print(f"Prix Heston semi-analytique: {sa_price:.4f} (temps: {sa_time:.2f}s)")
    
    # 3. Heston Monte Carlo pricing
    start_time = time()
    mc_pricer = MonteCarloPricer(european_option, heston_params, nb_paths=10000, nb_steps=100)
    mc_price = mc_pricer.price(random_seed=42)
    mc_time = time() - start_time
    print(f"Prix Heston Monte Carlo: {mc_price:.4f} (temps: {mc_time:.2f}s)")

    # Graphs commentés
    
    # # Test de sensibilité en fonction de la volatilité initiale
    # v0_range = np.linspace(0.01, 0.25, 10)
    # sa_prices = []
    # mc_prices = []
    
    # for v in v0_range:
    #     # Mise à jour des paramètres de Heston
    #     heston_params.v0 = v
        
    #     # Semi-analytique
    #     sa_pricer = SemiAnalyticalPricer(european_option, heston_params)
    #     sa_prices.append(sa_pricer.price())
        
    #     # Monte Carlo
    #     mc_pricer = MonteCarloPricer(european_option, heston_params, nb_paths=5000, nb_steps=50)
    #     mc_prices.append(mc_pricer.price(random_seed=42))
    
    # # Graphique des prix en fonction de la volatilité initiale
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.sqrt(v0_range), sa_prices, 'b-', label='Semi-analytique')
    # plt.plot(np.sqrt(v0_range), mc_prices, 'r--', label='Monte Carlo')
    # plt.xlabel('Volatilité initiale')
    # plt.ylabel('Prix de l\'option')
    # plt.title('Prix de l\'option européenne en fonction de la volatilité initiale')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('european_option_price_vs_vol.png')
    # plt.close()
    
    # # Test du delta en fonction du prix spot
    # spot_range = np.linspace(80, 120, 10)
    # deltas = []
    
    # for spot in spot_range:
    #     european_option.spot_price = spot
    #     delta = sa_pricer.first_order_derivative('spot_price', 0.01)
    #     deltas.append(delta)
    
    # # Graphique des deltas en fonction du prix spot
    # plt.figure(figsize=(10, 6))
    # plt.plot(spot_range, deltas, 'g-')
    # plt.xlabel('Prix spot')
    # plt.ylabel('Delta')
    # plt.title('Delta de l\'option européenne en fonction du prix spot')
    # plt.grid(True)
    # plt.savefig('european_option_delta.png')
    # plt.close()

def test_asian_option():
    """Test pricing d'une option asiatique avec Monte Carlo"""
    print("\n===== TEST OPTION ASIATIQUE =====")
    
    # Paramètres de l'option
    spot_price = 100.0
    strike = 100.0
    maturity = 1.0
    risk_free_rate = 0.05
    is_call = True
    
    # Création de l'option asiatique
    asian_option = AsianOption(spot_price, strike, maturity, risk_free_rate, is_call)
    
    # Paramètres du modèle de Heston
    kappa = 1.0
    theta = 0.04
    v0 = 0.04
    sigma = 0.2
    rho = -0.7
    
    heston_params = HestonParameters(kappa, theta, v0, sigma, rho)
    
    # Pricing Monte Carlo
    start_time = time()
    mc_pricer = MonteCarloPricer(asian_option, heston_params, nb_paths=50000, nb_steps=100)
    mc_price = mc_pricer.price(random_seed=42)
    mc_time = time() - start_time
    print(f"Prix option asiatique (Monte Carlo): {mc_price:.4f} (temps: {mc_time:.2f}s)")
    
    # Graphs commentés
#     # Test de prix en fonction du strike
#     strike_range = np.linspace(80, 120, 10)
#     mc_prices = []
    
#     for strike in strike_range:
#         asian_option.strike = strike
#         mc_pricer = MonteCarloPricer(asian_option, heston_params, nb_paths=10000, nb_steps=50)
#         mc_prices.append(mc_pricer.price(random_seed=42))
    
#     # Graphique des prix en fonction du strike
#     plt.figure(figsize=(10, 6))
#     plt.plot(strike_range, mc_prices, 'b-')
#     plt.xlabel('Strike')
#     plt.ylabel('Prix de l\'option')
#     plt.title('Prix de l\'option asiatique en fonction du strike')
#     plt.grid(True)
#     plt.savefig('asian_option_price_vs_strike.png')
#     plt.close()
    
#     # Test de price multiple pour obtenir un intervalle de confiance
#     asian_option.strike = 100.0
#     mc_pricer = MonteCarloPricer(asian_option, heston_params, nb_paths=5000, nb_steps=50)
#     start_time = time()
#     price_info = mc_pricer.price_multiple(10)
#     mc_time = time() - start_time
    
#     print(f"Prix moyen: {price_info[0]:.4f}")
#     print(f"Marge d'erreur (95%): {price_info[1]:.4f}")
#     print(f"Écart-type: {price_info[2]:.4f}")
#     print(f"Intervalle de confiance (95%): [{price_info[0] - price_info[1]:.4f}, {price_info[0] + price_info[1]:.4f}]")
#     print(f"Temps de calcul: {mc_time:.2f}s")

def compare_european_asian():
    """Comparaison des options européennes et asiatiques"""
    print("\n===== COMPARAISON EUROPÉENNE VS ASIATIQUE =====")
    
    # Paramètres communs
    spot_price = 100.0
    maturity = 1.0
    risk_free_rate = 0.05
    is_call = True
    
    # Paramètres du modèle de Heston
    kappa = 1.0
    theta = 0.04
    v0 = 0.04
    sigma = 0.2
    rho = -0.7
    
    heston_params = HestonParameters(kappa, theta, v0, sigma, rho)
    
    # Comparison en fonction du strike
    strike_range = np.linspace(80, 120, 10)
    european_prices = []
    asian_prices = []
    
    for strike in strike_range:
        # Option européenne
        euro_option = EuropeanOption(spot_price, strike, maturity, risk_free_rate, is_call)
        sa_pricer = SemiAnalyticalPricer(euro_option, heston_params)
        european_prices.append(sa_pricer.price())
        
        # Option asiatique
        asian_option = AsianOption(spot_price, strike, maturity, risk_free_rate, is_call)
        mc_pricer = MonteCarloPricer(asian_option, heston_params, nb_paths=10000, nb_steps=50)
        asian_prices.append(mc_pricer.price(random_seed=42))
    
    # Graphique de comparaison
    plt.figure(figsize=(10, 6))
    plt.plot(strike_range, european_prices, 'b-', label='Européenne')
    plt.plot(strike_range, asian_prices, 'r--', label='Asiatique')
    plt.xlabel('Strike')
    plt.ylabel('Prix de l\'option')
    plt.title('Comparaison des prix d\'options européennes et asiatiques')
    plt.legend()
    plt.grid(True)
    plt.savefig('european_vs_asian.png')
    plt.close()
    
    print("Différence de prix (Européenne - Asiatique):")
    for i, strike in enumerate(strike_range):
        diff = european_prices[i] - asian_prices[i]
        print(f"Strike {strike:.1f}: {diff:.4f}")

def test_pnl_matrix():
    """Test de la matrice de P&L"""
    print("\n===== TEST MATRICE DE P&L =====")
    
    # Paramètres de l'option
    spot_price = 100.0
    strike = 100.0
    maturity = 1.0
    risk_free_rate = 0.05
    is_call = True
    
    # Création de l'option européenne
    european_option = EuropeanOption(spot_price, strike, maturity, risk_free_rate, is_call)
    
    # Paramètres du modèle de Heston
    kappa = 1.0
    theta = 0.04
    v0 = 0.04
    sigma = 0.2
    rho = -0.7
    
    heston_params = HestonParameters(kappa, theta, v0, sigma, rho)
    
    # Pricer Monte Carlo
    mc_pricer = MonteCarloPricer(european_option, heston_params, nb_paths=10000, nb_steps=100)
    
    # Classe d'analyse
    analyzer = AnalysisTools(european_option, mc_pricer, heston_params)
    
    # Les plages de valeurs à tester
    spot_range = np.linspace(90, 110, 3)
    vol_range = np.linspace(0.02, 0.06, 3)
    
    # Calculer la matrice de P&L
    print("Calcul de la matrice de P&L...")
    start_time = time()
    pnl_matrix = analyzer.compute_pnl_matrix(
        x_param_name='spot_price',
        x_param_values=spot_range,
        y_param_name='v0',
        y_param_values=vol_range
    )
    calc_time = time() - start_time
    print(f"Temps de calcul: {calc_time:.2f}s")
    
    print("\nAperçu de la matrice de P&L:")
    print(pnl_matrix.round(4))
    
    plt.figure(figsize=(12, 8))
    analyzer.plot_pnl_matrix(pnl_matrix, title="Matrice de P&L - Option Européenne (Heston)")
    plt.savefig('pnl_matrix_european.png')
    plt.close()

def test_stress_scenarios():
    """Test des scénarios de stress"""
    print("\n===== TEST SCÉNARIOS DE STRESS =====")
    
    # Paramètres de l'option
    spot_price = 100.0
    strike = 100.0
    maturity = 1.0
    risk_free_rate = 0.05
    is_call = True
    
    # Création de l'option européenne
    european_option = EuropeanOption(spot_price, strike, maturity, risk_free_rate, is_call)
    
    # Paramètres du modèle de Heston
    kappa = 1.0
    theta = 0.04
    v0 = 0.04
    sigma = 0.2
    rho = -0.7
    
    heston_params = HestonParameters(kappa, theta, v0, sigma, rho)
    
    # Pricer Monte Carlo
    mc_pricer = MonteCarloPricer(european_option, heston_params, nb_paths=10000, nb_steps=100)
    
    # Classe d'analyse
    analyzer = AnalysisTools(european_option, mc_pricer, heston_params)
    
    # Définir des scénarios de stress
    crisis_scenarios = [
        # Conditions normales (notre référence)
        {'spot_price': 100.0, 'v0': 0.04, 'rho': -0.7},
        
        # Crash 2008
        {'spot_price': 70.0, 'v0': 0.25, 'rho': -0.9},
        
        # COVID-19 Mars 2020
        {'spot_price': 65.0, 'v0': 0.35, 'rho': -0.85},
        
        # Bulle Internet 2000
        {'spot_price': 80.0, 'v0': 0.20, 'rho': -0.75},
        
        # Hausse des taux 2022
        {'spot_price': 90.0, 'v0': 0.15, 'risk_free_rate': 0.08, 'rho': -0.6},
    ]
    
    scenario_names = [
        "Conditions normales",
        "Crise 2008",
        "COVID-19 (Mars 2020)",
        "Bulle Internet 2000",
        "Hausse des taux 2022"
    ]
    
    print("Exécution des scénarios de stress...")
    stress_results = analyzer.stress_test(crisis_scenarios, scenario_names)
    
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\nRésultats des scénarios de stress:")
    print(stress_results)
    
    # Graphique des résultats
    plt.figure(figsize=(12, 6))
    plt.bar(stress_results['Scenario'], stress_results['P&L (%)'])
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Scénario')
    plt.ylabel('P&L Relatif (%)')
    plt.title('Impact des scénarios de stress sur le P&L')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('stress_test_results.png')
    plt.close()



def main():
    print("==== TEST DU PRICER D'OPTIONS ====")
    
    #test_european_option()
    #test_asian_option()
    #compare_european_asian()

    print("==== TEST DES OUTILS D'ANALYSE D'OPTIONS ====")

    #test_pnl_matrix()
    #test_stress_scenarios()

    print("\nTous les tests sont terminés.")

if __name__ == "__main__":
    main()
