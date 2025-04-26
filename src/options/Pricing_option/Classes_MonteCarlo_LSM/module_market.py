# =====================
# module_marche.py
# =====================
from datetime import date

class MarketEnvironment:
    """
    Ici je définis tous les paramètres liés au marché :
    - prix spot
    - taux sans risque
    - volatilité
    - dividendes (à la fois continus et discrets)
    """

    def __init__(self, spot_price, risk_free_rate, volatility, dividend_yield=0.0):
        self.spot = spot_price              # S0 : Prix spot de l'actif
        self.rate = risk_free_rate          # r : taux sans risque
        self.vol = volatility               # sigma : volatilité
        self.q = dividend_yield             # q : taux de dividendes continus

        # Gestion des dividendes discrets
        self.dividend_dates = []            # Liste qui stocke les dates de paiement des dividendes
        self.dividends_flow = []            # Montant des dividendes associés
        self.dividend_steps = []            # Steps correspondants (calculés plus tard)

    def has_dividend(self, step: int) -> bool:
        """
        Vérifie s'il y a un dividende au step t
        """
        return step in self.dividend_steps

    def get_dividend(self, step: int) -> float:
        """
        Retourne le montant du dividende si jamais il tombe au step t
        """
        idx = self.dividend_steps.index(step)
        return self.dividends_flow[idx]