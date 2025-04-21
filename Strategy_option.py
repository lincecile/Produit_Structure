
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import List, Dict, Tuple, Union, Optional
from Pricing_option.Classes_MonteCarlo_LSM.module_LSM import LSM_method
from Pricing_option.Classes_Both.module_option import Option
from Pricing_option.Classes_Both.module_marche import DonneeMarche  
from Pricing_option.Classes_MonteCarlo_LSM.module_brownian import Brownian
import datetime as dt
from Pricing_option.Classes_Both.derivatives import OptionDerivatives

from Strategies_optionnelles.Portfolio_options import OptionsPortfolio
from Strategies_optionnelles.Strategies_predefinies import OptionsStrategy


date_pricing = dt.date(2024, 1, 13)
spot = 100
volatite = 0.2
risk_free_rate = 0.02
dividende_ex_date = dt.date.today()
dividende_montant = 0
maturite = dt.date(2024, 10, 23)
strike = 100
option_exercice = 'Européenne' 
option_type = "Call"
convention_base_calendaire = 365
parametre_alpha = 3
pruning = True
epsilon_arbre = 1e-15
nb_pas = 100
nb_chemin = 100000
seed_choice = 42
antithetic_choice = False
poly_degree = 2
regress_method = "Polynomial"
donnee_marche = DonneeMarche(date_pricing, spot, volatite, risk_free_rate, risk_free_rate, dividende_ex_date, dividende_montant)
option = Option(maturite, strike, barriere=None, 
            americaine=False if option_exercice == 'Européenne' else True, call=True if option_type == "Call" else False,
            date_pricing=date_pricing)

option1 = Option(maturite, strike, barriere=None, 
            americaine=False if option_exercice == 'Européenne' else True, call=False if option_type == "Call" else False,
            date_pricing=date_pricing)

option2 = Option(maturite, 50, barriere=None, 
            americaine=False if option_exercice == 'Européenne' else True, call=False if option_type == "Call" else False,
            date_pricing=date_pricing)

brownian = Brownian(time_to_maturity=(maturite-date_pricing).days / convention_base_calendaire, nb_step=nb_pas, nb_trajectoire=nb_chemin, seed=seed_choice)

portfolio = OptionsPortfolio(brownian,donnee_marche)

# Ajout d'options pour former un butterfly
portfolio.add_option(option1, 1)    # Long call bas strike
portfolio.add_option(option, 2)   # Short 2 calls strike moyen
portfolio.add_option(option, 1)
portfolio.add_option(option2, -1) 
portfolio.add_option(option2, -1) 
# portfolio.add_option(option, 2)    # Long call haut strike
print(portfolio.get_portfolio_detail())
portfolio.remove_option_quantity(1,1)
print(portfolio.get_portfolio_detail())
# Créer un gestionnaire de stratégies
# strategy = OptionsStrategy(portfolio, donnee_marche, expiry_date=maturite)

# Ajouter un bull call spread
# strategy.call_spread(lower_strike=100, upper_strike=110, quantity=1.0)

# Ajouter un straddle
# strategy.straddle(strike=105, quantity=1.0)

# Visualiser le payoff de la stratégie complète
# fig_portfolio = portfolio.plot_portfolio_payoff(show_individual=True)
# fig_option = portfolio.plot_option_payoff(1)

# Afficher les figures Plotly
# fig_portfolio.show()
# fig_option.show()
# # Calcul et affichage des grecques du portefeuille
# portfolio_greeks = portfolio.calculate_portfolio_greeks()
# print(portfolio_greeks)
# print("Grecques du portefeuille:")
# for greek, value in portfolio_greeks.items():
#     print(f"{greek.capitalize()}: {value:.6f}")

# # # Obtenir le résumé du portefeuille
# summary = portfolio.get_portfolio_summary()
# print("\nRésumé du portefeuille:")
# for key, value in summary.items():
#     print(f"{key}: {value}")

# Tracer le payoff
# fig_payoff, ax_payoff = portfolio.plot_payoff(include_individual_options=True)

# # Tracer les grecques (réduit le nombre de points pour accélérer)
# spot_range = np.linspace(80, 120, 10)
# fig_greeks, axes_greeks = portfolio.plot_greeks_vs_spot(spot_range=spot_range)

# plt.show()