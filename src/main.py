# =====================
# main_pricer.py
# =====================
from module_option import Option
from module_market import MarketEnvironment
from module_brownian import Brownian
from module_lsm_barrier import LSM

import datetime as dt

# 1. Définir l'option
option = Option(
    call=True,                 # Option Call
    prix_exercice=100,          # Strike 100
    maturity=1.0,              # 1 an
    americaine=True,            # Américaine (mettre False pour Européenne)
    barriere=120,               # Barrière (exemple up-and-out à 120)
    type_barriere="up-out",    # Type de barrière
    date_pricing=dt.date.today()
)

# 2. Définir le marché
market = MarketEnvironment(
    spot_price=100,             # Spot initial 100
    risk_free_rate=0.02,        # Taux sans risque 2%
    volatility=0.2,             # Volatilité 20%
    dividend_yield=0.0          # Pas de dividende continu ici
)

# Exemple si tu veux ajouter un dividende discret :
# market.register_dividend(pay_date=dt.date.today() + dt.timedelta(days=180), amount=2.0)

# 3. Définir le Brownien
brownian = Brownian(
    time_to_maturity=option.maturity,
    nb_step=50,                 # Nombre de steps dans la simulation
    nb_trajectoire=10000,       # Nombre de trajectoires
    seed=42
)

# 4. Pricer l'option
pricer = LSM(option)
prix, ecart_type, intervalle_confiance = pricer.price(
    brownian=brownian,
    market=market,
    poly_degree=2,
    model_type="Polynomial",
    antithetic=True
)

# 5. Résultat
print(f"Prix estimé : {prix:.4f}")
print(f"Écart-type : {ecart_type:.4f}")
print(f"Intervalle de confiance 95% : [{intervalle_confiance[0]:.4f}, {intervalle_confiance[1]:.4f}]")

