#%% Imports
import concurrent.futures
import time
import pandas as pd
import datetime as dt
import sys
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

sys.setrecursionlimit(1000000000)

# from module_marche import DonneeMarche
# from module_option import Option
# from module_barriere import Barriere
# from module_arbre_noeud import Arbre

# # import des fonctions Black Scholes
# from module_black_scholes import BlackAndScholes

from Pricing_option.Classes_Both.module_marche import DonneeMarche
from Pricing_option.Classes_Both.module_option import Option
from Pricing_option.Classes_TrinomialTree.module_barriere import Barriere
from Pricing_option.Classes_TrinomialTree.module_arbre_noeud import Arbre

# import des fonctions Black Scholes
from Pricing_option.Classes_Both.module_black_scholes import BlackAndScholes
from Pricing_option.Classes_Both.module_black_scholes import BlackAndScholes

from Pricing_option.Classes_MonteCarlo_LSM.module_brownian import Brownian
from Pricing_option.Classes_MonteCarlo_LSM.module_LSM import LSM_method
from copy import deepcopy
 
#%% Classes

class BsComparison:
    
    def __init__(self, max_cpu : int,  step_list : list, epsilon_values : list):
        
        self.max_cpu = max_cpu
        
        # Definissons les deux listes
        self.step_list = step_list
        self.epsilon_values = epsilon_values

        # Instanciation des objets requis
        self.barriere = Barriere(0, None, None)
        self.donnee_marche = DonneeMarche(dt.date(2024, 1, 13), 100, 0.20, 0.02, 0.02, dt.date.today(), 0)
        self.option = Option(dt.date(2024, 10, 23), 101, self.barriere, False, True, dt.date(2024, 1, 13))
        
        # Calcul du prix B&S
        self.arbre_bs = Arbre(100, self.donnee_marche, self.option)
        self.bs_price = BlackAndScholes(self.arbre_bs).bs_pricer()

        # DataFrame pour stocker les résultats
        self.results_df = pd.DataFrame()

        # Executions des calculs en parallèle pour chaque niveau d'epsilon
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_cpu) as outer_executor:
            futures = {outer_executor.submit(self.calculate_for_epsilon, epsilon): epsilon for epsilon in self.epsilon_values}

            # Résultats
            for future in concurrent.futures.as_completed(futures):
                epsilon = futures[future]
                try:
                    result_df = future.result()
                    result_df['Epsilon (1e-)'] = epsilon
                    self.results_df = pd.concat([self.results_df, result_df], ignore_index=True)
                except Exception as exc:
                    print(f'Erreur à epsilon {epsilon} : {exc}')

    def calculate_for_epsilon(self, epsilon):
        liste_prix_step_comparison = []
        liste_temps_step_comparison = []
        liste_diff_bs = []
        liste_time_gap_bs = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_cpu) as inner_executor:
            futures = {
                inner_executor.submit(self.calcule_pas, step, self.bs_price, self.donnee_marche, self.option, epsilon): step 
                for step in self.step_list
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                liste_prix_step_comparison.append(result[1])
                liste_temps_step_comparison.append(result[2])
                liste_diff_bs.append(result[3])
                liste_time_gap_bs.append(result[4])

        
        return pd.DataFrame({
            "Nombre de pas": self.step_list,
            "Prix arbre trinomial": liste_prix_step_comparison,
            "Temps pricing": liste_temps_step_comparison,
            "Différence B&S": liste_diff_bs,
            "Différence * nb pas": liste_time_gap_bs,
        })

    @staticmethod
    def calcule_pas(step, bs_price, donnee_marche, option, epsilon):
        now = time.time()
        arbre_step_comparison = Arbre(step, donnee_marche, option, epsilon=epsilon)
        arbre_step_comparison.pricer_arbre()
        price_arbre_step_comparison = arbre_step_comparison.prix_option
        then = time.time()
        pricing_time = then - now
        print(f"Step: {step}, Epsilon: {epsilon}, Pricing Time: {pricing_time:.4f}s")
        return (step, price_arbre_step_comparison, pricing_time, 
                price_arbre_step_comparison - bs_price, 
                price_arbre_step_comparison * step)    
        
    def bs_graph_temps_pas(self): 
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.results_df["Nombre de pas"],
            y=self.results_df["Temps pricing"],
            mode='lines+markers',
            name='Temps de pricing',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title='Temps de pricing en fonction du nombre de pas',
            xaxis_title='Nombre de pas',
            yaxis_title='Temps de pricing (secondes)',
        )
        
        return fig
    
    def bs_graph_prix_pas(self): 
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.results_df["Nombre de pas"],
            y=self.results_df["Prix arbre trinomial"],
            mode='lines+markers',
            name='Prix arbre trinomial',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_hline(
        y=self.bs_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Prix B&S: {round(self.bs_price,4)}",
        annotation_position="bottom right",
        annotation_font=dict(size=12, color="red")
        )

        fig.update_layout(
            title='''Prix renvoyé par l'arbre en fonction du nombre de pas''',
            xaxis_title='Nombre de pas',
            yaxis_title='Prix arbre',
        )
        
        return fig
    
    def epsilon_graph_prix_pas_bas_epsilon(self):
        
        fig = go.Figure()

        mask = [
            (value[0] < 1e-7) if isinstance(value, list) and len(value) > 0 else (value < 1e-7)
            for value in self.results_df['Epsilon (1e-)']
        ]

        filtered_df = self.results_df[mask]
        
        unique_epsilons = filtered_df['Epsilon (1e-)'].unique()

        for epsilon in unique_epsilons:
            filtered_data = self.results_df[self.results_df['Epsilon (1e-)'] == epsilon]
            fig.add_trace(go.Scatter(
                x=filtered_data["Nombre de pas"],
                y=filtered_data["Prix arbre trinomial"],
                mode='lines+markers',
                name=f'Epsilon = {epsilon}',
                line=dict(width=2),
                marker=dict(size=6)
            ))
            
        fig.add_hline(
            y=self.bs_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Prix B&S: {round(self.bs_price, 4)}",
            annotation_position="bottom right",
            annotation_font=dict(size=12, color="red")
        )

        fig.update_layout(
            title='''Prix renvoyé par l'arbre en fonction du nombre de pas''',
            xaxis_title='Nombre de pas',
            yaxis_title='Prix arbre',
            legend_title='Epsilon Values'
        )

        return fig

    def epsilon_graph_prix_pas_haut_epsilon(self):
        
        fig = go.Figure()

        mask = [
            (value[0] > 1e-7) if isinstance(value, list) and len(value) > 0 else (value > 1e-7)
            for value in self.results_df['Epsilon (1e-)']
        ]

        filtered_df = self.results_df[mask]
        
        unique_epsilons = filtered_df['Epsilon (1e-)'].unique()

        for epsilon in unique_epsilons:
            filtered_data = self.results_df[self.results_df['Epsilon (1e-)'] == epsilon]
            fig.add_trace(go.Scatter(
                x=filtered_data["Nombre de pas"],
                y=filtered_data["Prix arbre trinomial"],
                mode='lines+markers',
                name=f'Epsilon = {epsilon}',
                line=dict(width=2),
                marker=dict(size=6)
            ))
            
        fig.add_hline(
            y=self.bs_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Prix B&S: {round(self.bs_price, 4)}",
            annotation_position="bottom right",
            annotation_font=dict(size=12, color="red")
        )

        fig.update_layout(
            title='''Prix renvoyé par l'arbre en fonction du nombre de pas''',
            xaxis_title='Nombre de pas',
            yaxis_title='Prix arbre',
            legend_title='Epsilon Values'
        )

        return fig

    def epsilon_vs_temps_pricing_graph(self):
        fig = go.Figure()

        mask = [
            (value[0] == 5000) if isinstance(value, list) and len(value) > 0 else (value == 5000)
            for value in self.results_df['Nombre de pas']
        ]

        filtered_df = self.results_df[mask]

        fig.add_trace(go.Scatter(
            x=filtered_df["Epsilon (1e-)"],
            y=filtered_df["Temps pricing"],
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6)
        ))

        fig.update_xaxes(type='log', title='Epsilon', tickformat='1e', autorange='reversed')

        fig.update_layout(
            title='Temps de valorisation pour un arbre de 5000 pas',
            xaxis_title='Epsilon',
            yaxis_title='Temps de valorisation (secondes)',
        )

        return fig
    
    
class StrikeComparison:
    
    def __init__(self, max_cpu: int, step_list: list, strike_values: list,
                 donnee_marche: DonneeMarche, brownian: Brownian, option: Option):
        self.max_cpu = max_cpu
        self.step_list = step_list
        self.strike_values = strike_values
        self.donnee_marche = donnee_marche
        self.brownian = brownian
        self.option = option

        self.barriere = Barriere(0, None, None)
        self.results_df = pd.DataFrame()

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_cpu) as outer_executor:
            futures = {
                outer_executor.submit(self.calculate_for_strike, strike): strike
                for strike in self.strike_values
            }

            for future in concurrent.futures.as_completed(futures):
                strike = futures[future]
                try:
                    result_df = future.result()
                    result_df['Strike'] = strike
                    self.results_df = pd.concat([self.results_df, result_df], ignore_index=True)
                except Exception as exc:
                    print(f'Error at strike {strike}: {exc}')

    def calculate_for_strike(self, strike):
        option = Option(
            date_pricing=self.option.date_pricing,
            maturite=self.option.maturite,
            prix_exercice=strike,
            barriere=self.barriere,
            americaine=False,
            call=True  # ou True selon le besoin
        )

        arbre_bs = Arbre(100, self.donnee_marche, option)
        bs_price = BlackAndScholes(arbre_bs).bs_pricer()

        step_results = []

        for step in self.step_list:
            res = self.calcule_pas(step, bs_price, self.donnee_marche, option)
            step_results.append(res)

        return pd.DataFrame(step_results)

    def calcule_pas(self, step, bs_price, donnee_marche, option):
        # Prix arbre trinomial
        now_tree = time.time()
        arbre = Arbre(100, donnee_marche, option)
        arbre.pricer_arbre()
        prix_arbre = arbre.prix_option
        time_tree = time.time() - now_tree

        # Prix LSM
        now_lsm = time.time()
        pricer_lsm = LSM_method(option)
        prix_lsm, std_err, _ = pricer_lsm.LSM(
            self.brownian, donnee_marche,
            method='vector')
        time_lsm = time.time() - now_lsm
        print('strike en cours:', option.prix_exercice)
        return {
            "Nombre de pas": step,
            "Prix arbre trinomial": prix_arbre,
            "Temps arbre": time_tree,
            "Diff arbre - B&S": prix_arbre - bs_price,

            "Prix LSM": prix_lsm,
            "Temps LSM": time_lsm,
            "Diff LSM - B&S": prix_lsm - bs_price,

            "Diff arbre * nb pas": (prix_arbre - bs_price) * 100,
            "Diff LSM * nb pas": (prix_lsm - bs_price) * step,
        }

    def graph_strike_comparison(self):
        fig = go.Figure()
        sorted_df = self.results_df.sort_values('Strike')

        for step in self.step_list:
            df_step = sorted_df#[sorted_df['Strike'] == step]

            fig.add_trace(go.Scatter(
                x=df_step["Strike"],
                y=df_step["Diff arbre - B&S"],
                mode='lines+markers',
                name=f'Arbre (pas={100})',
                line=dict(dash='solid')
            ))

            fig.add_trace(go.Scatter(
                x=df_step["Strike"],
                y=df_step["Diff LSM - B&S"],
                mode='lines+markers',
                name=f'LSM (pas={step})',
                line=dict(dash='dot')
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="red")

        fig.update_layout(
            title='Comparaison (Arbre vs LSM) avec Black-Scholes en fonction du Strike',
            xaxis_title='Strike',
            yaxis_title='Différence par rapport à Black-Scholes',
            legend_title='Méthodes'
        )

        return fig
    
class VolComparison:
    
    def __init__(self, max_cpu: int, step_list: list, vol_values: list,
                 donnee_marche: DonneeMarche, brownian: Brownian, option: Option):
        self.max_cpu = max_cpu
        self.step_list = step_list
        self.vol_values = vol_values
        self.donnee_marche = donnee_marche
        self.brownian = brownian
        self.option = option

        self.barriere = Barriere(0, None, None)
        self.results_df = pd.DataFrame()

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_cpu) as outer_executor:
            futures = {
                outer_executor.submit(self.calculate_for_vol, vol): vol
                for vol in self.vol_values
            }

            for future in concurrent.futures.as_completed(futures):
                vol = futures[future]
                try:
                    result_df = future.result()
                    result_df['Volatilité'] = vol
                    self.results_df = pd.concat([self.results_df, result_df], ignore_index=True)
                except Exception as exc:
                    print(f'Error at vol {vol}: {exc}')

    def calculate_for_vol(self, vol):
        # Met à jour la volatilité du marché
        donnee_marche = DonneeMarche(
            date_debut=self.option.date_pricing,
            prix_spot=self.donnee_marche.prix_spot,
            volatilite=vol,
            taux_interet=self.donnee_marche.taux_interet,
            taux_actualisation=self.donnee_marche.taux_actualisation,
            dividende_ex_date=self.donnee_marche.dividende_ex_date,
            dividende_montant=self.donnee_marche.dividende_montant,
            dividende_rate=self.donnee_marche.dividende_rate
        )

        # Calcul Black-Scholes avec l'arbre trinomial
        arbre_bs = Arbre(100, donnee_marche, self.option)
        bs_price = BlackAndScholes(arbre_bs).bs_pricer()

        step_results = []

        for step in self.step_list:
            res = self.calcule_pas(step, bs_price, donnee_marche, self.option)
            step_results.append(res)

        return pd.DataFrame(step_results)

    def calcule_pas(self, step, bs_price, donnee_marche, option):
        # Prix arbre trinomial
        now_tree = time.time()
        arbre = Arbre(100, donnee_marche, option)
        arbre.pricer_arbre()
        prix_arbre = arbre.prix_option
        time_tree = time.time() - now_tree

        # Prix LSM
        now_lsm = time.time()
        pricer_lsm = LSM_method(option)
        prix_lsm, std_err, _ = pricer_lsm.LSM(
            self.brownian, donnee_marche,
            method='vector')
        time_lsm = time.time() - now_lsm
        print('Volatilité en cours:', donnee_marche.volatilite)
        
        return {
            "Nombre de pas": step,
            "Prix arbre trinomial": prix_arbre,
            "Temps arbre": time_tree,
            "Diff arbre - B&S": prix_arbre - bs_price,

            "Prix LSM": prix_lsm,
            "Temps LSM": time_lsm,
            "Diff LSM - B&S": prix_lsm - bs_price,

            "Diff arbre * nb pas": (prix_arbre - bs_price) * 100,
            "Diff LSM * nb pas": (prix_lsm - bs_price) * step,
        }

    def graph_vol(self):
        fig = go.Figure()
        sorted_df = self.results_df.sort_values('Volatilité')

        for step in self.step_list:
            df_step = sorted_df[sorted_df['Nombre de pas'] == step]

            fig.add_trace(go.Scatter(
                x=df_step["Volatilité"],
                y=df_step["Diff arbre - B&S"],
                mode='lines+markers',
                name=f'Arbre (pas={100})',
                line=dict(dash='solid')
            ))

            fig.add_trace(go.Scatter(
                x=df_step["Volatilité"],
                y=df_step["Diff LSM - B&S"],
                mode='lines+markers',
                name=f'LSM (pas={step})',
                line=dict(dash='dot')
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="red")

        fig.update_layout(
            title='Comparaison (Arbre vs LSM) avec Black-Scholes en fonction de la Volatilité',
            xaxis_title='Volatilité',
            yaxis_title='Différence par rapport à Black-Scholes',
            legend_title='Méthodes'
        )

        return fig

class RateComparison:
    
    def __init__(self, max_cpu: int, step_list: list, rate_values: list, donnee_marche, brownian=None, option=None):
        self.max_cpu = max_cpu
        self.step_list = step_list
        self.rate_values = rate_values
        self.brownian = brownian  # Ajout du paramètre brownian pour LSM
        self.donnee_marche = donnee_marche
        self.barriere = Barriere(0, None, None)
        self.option = option
        self.results_df = pd.DataFrame()

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_cpu) as outer_executor:
            futures = {outer_executor.submit(self.calculate_for_rate, rate): rate for rate in self.rate_values}

            for future in concurrent.futures.as_completed(futures):
                rate = futures[future]
                try:
                    result_df = future.result()
                    result_df['Taux d\'intérêt'] = rate
                    self.results_df = pd.concat([self.results_df, result_df], ignore_index=True)
                except Exception as exc:
                    print(f'Error at rate {rate}: {exc}')

    def calculate_for_rate(self, rate):

        donnee_marche = DonneeMarche(
            date_debut=self.donnee_marche.date_debut,
            prix_spot=self.donnee_marche.prix_spot,
            volatilite=self.donnee_marche.volatilite,
            taux_interet=rate,
            taux_actualisation=rate,
            dividende_ex_date=self.donnee_marche.dividende_ex_date,
            dividende_montant=self.donnee_marche.dividende_montant,
            dividende_rate=self.donnee_marche.dividende_rate
        )

        self.arbre_bs = Arbre(100, donnee_marche, self.option)
        self.bs_price = BlackAndScholes(self.arbre_bs).bs_pricer()

        liste_prix_step_comparison = []
        liste_temps_step_comparison = []
        liste_diff_bs_arbre = []
        liste_time_gap_bs_arbre = []
        
        # Ajout pour LSM
        liste_prix_lsm = []
        liste_temps_lsm = []
        liste_diff_bs_lsm = []
        liste_time_gap_bs_lsm = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_cpu) as inner_executor:
            futures = {
                inner_executor.submit(self.calcule_pas, step, self.bs_price, donnee_marche, self.option): step 
                for step in self.step_list
            }

            for future in concurrent.futures.as_completed(futures):
                step = futures[future]
                result = future.result()
                
                # Résultats de l'arbre
                liste_prix_step_comparison.append(result[0])
                liste_temps_step_comparison.append(result[1])
                liste_diff_bs_arbre.append(result[2])
                liste_time_gap_bs_arbre.append(result[3])
                
                # Résultats de LSM
                liste_prix_lsm.append(result[4])
                liste_temps_lsm.append(result[5])
                liste_diff_bs_lsm.append(result[6])
                liste_time_gap_bs_lsm.append(result[7])
        
        return pd.DataFrame({
            "Nombre de pas": self.step_list,
            "Prix arbre trinomial": liste_prix_step_comparison,
            "Temps arbre": liste_temps_step_comparison,
            "Diff arbre - B&S": liste_diff_bs_arbre,
            "Diff arbre * nb pas": liste_time_gap_bs_arbre,
            "Prix LSM": liste_prix_lsm,
            "Temps LSM": liste_temps_lsm,
            "Diff LSM - B&S": liste_diff_bs_lsm,
            "Diff LSM * nb pas": liste_time_gap_bs_lsm,
        })

    def calcule_pas(self, step, bs_price, donnee_marche, option):
        # Prix arbre trinomial
        now_tree = time.time()
        arbre_step_comparison = Arbre(step, donnee_marche, option)
        arbre_step_comparison.pricer_arbre()
        price_arbre_step_comparison = arbre_step_comparison.prix_option
        then_tree = time.time()
        tree_pricing_time = then_tree - now_tree
        
        # Prix LSM
        now_lsm = time.time()
        pricer_lsm = LSM_method(option)
        prix_lsm, std_err, _ = pricer_lsm.LSM(
            self.brownian, donnee_marche,
            method='vector')
        then_lsm = time.time()
        lsm_pricing_time = then_lsm - now_lsm
        
        print(f"Taux d'intérêt: {donnee_marche.taux_interet}, "
              f"Prix arbre: {price_arbre_step_comparison}, Temps arbre: {tree_pricing_time:.4f}s, "
              f"Prix LSM: {prix_lsm}, Temps LSM: {lsm_pricing_time:.4f}s")
        
        return (
            price_arbre_step_comparison,       # Prix arbre
            tree_pricing_time,                 # Temps arbre
            price_arbre_step_comparison - bs_price,  # Différence arbre - B&S
            (price_arbre_step_comparison - bs_price) * step,  # Différence * nb pas
            prix_lsm,                          # Prix LSM
            lsm_pricing_time,                  # Temps LSM
            prix_lsm - bs_price,               # Différence LSM - B&S
            (prix_lsm - bs_price) * step       # Différence LSM * nb pas
        )
        
    def graph_rate(self): 
        fig = go.Figure()
        
        sorted_df = self.results_df.sort_values('Taux d\'intérêt', ascending=True)

        # Tracer la courbe pour l'arbre trinomial
        fig.add_trace(go.Scatter(
            x=sorted_df['Taux d\'intérêt'],
            y=sorted_df['Diff arbre - B&S'],
            mode='lines+markers',
            name='Arbre trinomial',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Tracer la courbe pour LSM
        fig.add_trace(go.Scatter(
            x=sorted_df['Taux d\'intérêt'],
            y=sorted_df['Diff LSM - B&S'],
            mode='lines+markers',
            name='LSM',
            line=dict(color='green', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Ligne de référence à y=0
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red"
        )

        fig.update_layout(
            title='Comparaison (Arbre vs LSM) avec Black-Scholes en fonction du Taux',
            xaxis_title='Taux d\'intérêt',
            yaxis_title='Différence par rapport à Black-Scholes',
            legend_title='Méthodes'
        )
        
        return fig