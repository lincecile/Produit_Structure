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

    def register_dividend(self, pay_date: date, amount: float):
        """
        Cette fonction permet d'enregistrer un dividende à une date précise
        """
        self.dividend_dates.append(pay_date)
        self.dividends_flow.append(amount)

    def list_dividends(self):
        """
        Fonction juste pour afficher les dividendes enregistrés (utile pour vérifier)
        """
        print("Dividendes enregistrés :")
        for d, v in zip(self.dividend_dates, self.dividends_flow):
            print(f"Dividende de {v} versé le {d}")

    def compute_dividend_steps(self, start_date: date, maturity: float, nb_steps: int):
        """
        Convertit les dates réelles des dividendes → en steps (t)
        """
        dt = maturity / nb_steps
        self.dividend_steps = [
            int((div_date - start_date).days / 365 / dt)
            for div_date in self.dividend_dates
        ]

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