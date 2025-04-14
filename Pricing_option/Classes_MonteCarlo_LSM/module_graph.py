import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from matplotlib.gridspec import GridSpec
import seaborn as sns
from Classes_Both.module_option import Option
from Classes_Both.module_marche import DonneeMarche
from Classes_MonteCarlo_LSM.module_brownian import Brownian
from Classes_MonteCarlo_LSM.module_LSM import LSM_method
from copy import deepcopy


import plotly.graph_objects as go
from plotly.subplots import make_subplots


import numpy as np

class LSMGraph:
    """
    Classe pour la visualisation graphique des résultats des méthodes de pricing d'options
    par la méthode Least Square Monte Carlo (LSM).
    """
    
    def __init__(self, option : Option, market : DonneeMarche):
        """
        Initialisation de la classe LSMGraph.
        
        Args:
            option: Instance d'Option à évaluer
            market: Instance de DonneeMarche avec les paramètres du marché
        """
        self.option = deepcopy(option)
        self.market = deepcopy(market)
        self.period = (option.maturite - option.date_pricing).days / 365
        
        # Configuration des styles pour les graphiques
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.markers = ['o', 's', '^', 'D', 'x', '*', '+', 'v', '<', '>']
        self.line_styles = ['-', '--', '-.', ':']
        
        # Style Seaborn pour de meilleurs graphiques
        sns.set_style("whitegrid")
        self.palette = sns.color_palette("muted")
        

    def afficher_mouvements_browniens(self, brownian, nb_trajectoires=5):
        """
        Affiche les mouvements browniens pour un nombre limité de simulations.
        
        Args:
            brownian: Instance de Brownian
            nb_trajectoires: Nombre de trajectoires à afficher (défaut: 5)
            
        Returns:
            fig: Figure plotly
        """
        # Créer une figure plotly
        fig = go.Figure()
        
        times = np.linspace(0, self.period, brownian.nb_step + 1)
        
        # Générer tous les mouvements browniens en une seule fois
        brownian_paths = brownian.Vecteur()
        
        # Sélectionner aléatoirement nb_trajectoires parmi celles générées
        indices = np.random.choice(brownian_paths.shape[0], 
                                min(nb_trajectoires, brownian_paths.shape[0]), 
                                replace=False)
        
        for i, idx in enumerate(indices):
            color = self.colors[i % len(self.colors)]
            fig.add_trace(
                go.Scatter(
                    x=times, 
                    y=brownian_paths[idx, :], 
                    mode='lines', 
                    line=dict(color=color, width=2),
                    opacity=0.8,
                    name=f'Brownien {i+1}'
                )
            )
        
        # Ajouter la ligne horizontale à zéro
        fig.add_trace(
            go.Scatter(
                x=[0, self.period], 
                y=[0, 0], 
                mode='lines', 
                line=dict(color='black', width=2, dash='dash'),
                name="Niveau 0"
            )
        )
        
        # Mettre à jour la mise en page
        fig.update_layout(
            title="Simulation de mouvements browniens",
            xaxis_title="Temps (années)",
            yaxis_title="Valeur du mouvement brownien",
            legend_title="Trajectoires",
            template="plotly_white"
        )
        
        return fig

    def afficher_trajectoires_prix(self, trajectoires, brownian, nb_trajectoires=5):
        """
        Affiche les trajectoires de prix pour un nombre limité de simulations.
        
        Args:
            trajectoires: Array numpy contenant les trajectoires de prix
            brownian: Instance de Brownian utilisée pour générer les trajectoires
            nb_trajectoires: Nombre de trajectoires à afficher (défaut: 5)
            
        Returns:
            fig: Figure plotly
        """
        # Créer une figure plotly
        fig = go.Figure()
        
        # Obtenir les dates/temps et sélectionner des trajectoires aléatoires
        times = np.linspace(0, self.period, brownian.nb_step + 1)
        indices = np.random.choice(trajectoires.shape[0], min(nb_trajectoires, trajectoires.shape[0]), replace=False)
        
        for i, idx in enumerate(indices):
            color = self.colors[i % len(self.colors)]
            fig.add_trace(
                go.Scatter(
                    x=times, 
                    y=trajectoires[idx, :], 
                    mode='lines', 
                    line=dict(color=color, width=2),
                    opacity=0.8,
                    name=f'Trajectoire {i+1}'
                )
            )
        
        # Ajouter le prix spot initial
        fig.add_trace(
            go.Scatter(
                x=[0, self.period], 
                y=[self.market.prix_spot, self.market.prix_spot], 
                mode='lines', 
                line=dict(color='black', width=2, dash='dash'),
                name=f"Prix spot initial ({self.market.prix_spot:.2f})"
            )
        )
        
        # Ajouter le prix d'exercice
        fig.add_trace(
            go.Scatter(
                x=[0, self.period], 
                y=[self.option.prix_exercice, self.option.prix_exercice], 
                mode='lines', 
                line=dict(color='red', width=2, dash='dash'),
                name=f"Prix d'exercice ({self.option.prix_exercice:.2f})"
            )
        )
        
        # Mettre à jour la mise en page
        fig.update_layout(
            title="Simulation de trajectoires de prix",
            xaxis_title="Temps (années)",
            yaxis_title="Prix du sous-jacent",
            legend_title="Trajectoires",
            template="plotly_white"
        )
        
        return fig

    def comparer_methodes(self, methods=None, nb_paths=1000, nb_steps=200, seed=42):
        """
        Compare différentes méthodes de calcul du prix d'une option.
        
        Args:
            methods: Liste des méthodes à comparer ['vector', 'scalar']
            nb_paths: Nombre de chemins pour la simulation
            nb_steps: Nombre de pas de temps pour la simulation
            
        Returns:
            fig: Figure plotly
        """
        if methods is None:
            methods = ["vector", "scalar"]
            
        prices = []
        std_devs = []
        execution_times = []

        for method in methods:
            pricer = LSM_method(self.option)
            brownian = Brownian(self.period, nb_steps, nb_paths, seed)
            start_time = time.time()
            price, std_error, interval = pricer.LSM(brownian, self.market, method=method)
            end_time = time.time()
            execution_time = end_time - start_time
            
            execution_times.append(execution_time)
            prices.append(price)
            std_devs.append(std_error)

        # Création de la figure plotly
        fig = go.Figure()
        
        # Ajouter les barres d'erreur pour chaque méthode
        for i, method in enumerate(methods):
            fig.add_trace(go.Scatter(
                x=[method],
                y=[prices[i]],
                error_y=dict(
                    type='data',
                    array=[2 * std_devs[i]],
                    visible=True
                ),
                mode='markers',
                marker=dict(size=12, color=self.colors[i % len(self.colors)]),
                name=f"{method} ({execution_times[i]:.2f}s)"
            ))
        
        # Ajouter une ligne pour le prix moyen
        fig.add_shape(
            type="line",
            x0=-0.5, 
            y0=np.mean(prices), 
            x1=len(methods)-0.5, 
            y1=np.mean(prices),
            line=dict(
                color="black",
                width=2,
                dash="dash",
            )
        )
        
        # Ajouter une annotation pour le prix moyen
        fig.add_annotation(
            x=len(methods)-1,
            y=np.mean(prices),
            text=f"Prix moyen: {np.mean(prices):.4f}",
            showarrow=False,
            yshift=10
        )
        
        # Mise en page
        fig.update_layout(
            title="Intervalles de confiance des prix (± 2 écart-types)",
            xaxis_title="Méthodes",
            yaxis_title="Prix",
            template="plotly_white"
        )
        
        return fig
    
    def comparer_convergence_paths(self, reference_price=None, methods=None, nb_points=20, max_paths=10000):
        """
        Compare la convergence du prix en fonction du nombre de paths pour différentes méthodes.
        
        Args:
            reference_price: Prix de référence (ex: price_BS ou price_tree)
            methods: Liste des méthodes à comparer ['vector', 'scalar']
            nb_points: Nombre de points sur le graphique
            max_paths: Nombre maximum de paths
            
        Returns:
            fig: Figure plotly
        """
        if methods is None:
            methods = ["scalar", "vector"]

        # Générer une échelle logarithmique pour le nombre de paths
        paths = np.logspace(1, np.log10(max_paths), num=nb_points, dtype=int)
        
        results = {}
        for method in methods:
            method_prices = []
            method_std_devs = []
            method_times = []
            
            print(f"Traitement de la méthode: {method}")
            for path in paths:
                pricer = LSM_method(self.option)
                brownian = Brownian(self.period, 20, path, 1)
                
                start_time = time.time()
                price, std_error, _ = pricer.LSM(brownian, self.market, method=method)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                method_prices.append(price)
                method_std_devs.append(std_error)
                method_times.append(execution_time)
            
            results[method] = {
                'prices': method_prices,
                'std_devs': method_std_devs,
                'times': method_times
            }

        # Création de la figure plotly avec deux sous-tracés
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "Évolution du prix en fonction du nombre de chemins (± 2 écart-types)",
            "Temps d'exécution en fonction du nombre de chemins"
        ))
        
        for i, method in enumerate(methods):
            color = self.colors[i % len(self.colors)]
            
            # Calcul du temps moyen d'exécution
            avg_time = np.mean(results[method]['times'])
            
            # Graphique des prix avec barres d'erreur
            fig.add_trace(
                go.Scatter(
                    x=paths,
                    y=results[method]['prices'],
                    error_y=dict(
                        type='data',
                        array=2 * np.array(results[method]['std_devs']),
                        visible=True
                    ),
                    mode='lines+markers',
                    line=dict(color=color),
                    marker=dict(symbol=i, size=8),
                    name=f"{method} (Moy: {avg_time:.2f}s)",
                ),
                row=1, col=1
            )
            
            # Graphique des temps d'exécution
            fig.add_trace(
                go.Scatter(
                    x=paths,
                    y=results[method]['times'],
                    mode='lines+markers',
                    line=dict(color=color),
                    marker=dict(symbol=i, size=8),
                    name=f"{method}",
                    showlegend=False
                ),
                row=1, col=2
            )

        # Si un prix de référence est fourni
        if reference_price is not None:
            fig.add_shape(
                type="line",
                x0=paths[0], 
                y0=reference_price, 
                x1=paths[-1], 
                y1=reference_price,
                line=dict(
                    color="black",
                    width=2,
                    dash="dash",
                ),
                row=1, col=1
            )
            
            fig.add_annotation(
                x=paths[-1],
                y=reference_price,
                text=f"Prix de référence: {reference_price:.4f}",
                showarrow=False,
                yshift=10,
                row=1, col=1
            )

        # Mise à jour de la mise en page
        fig.update_xaxes(title_text="Nombre de chemins (échelle log)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Nombre de chemins (échelle log)", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Prix de l'option", row=1, col=1)
        fig.update_yaxes(title_text="Temps d'exécution (s)", type="log", row=1, col=2)
        
        fig.update_layout(
            height=600,
            width=1200,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def comparer_steps(self, nb_steps_list=None, nb_paths=10000, seed=42):
        """
        Compare les résultats pour différentes valeurs du nombre de pas (steps).
        
        Args:
            nb_steps_list: Liste des valeurs de pas à tester
            nb_paths: Nombre de chemins pour la simulation
            seed: Seed aléatoire fixe pour la reproductibilité
            
        Returns:
            fig: Figure plotly
        """
        if nb_steps_list is None:
            nb_steps_list = np.linspace(10, 400, num=200, dtype=int)  # Par défaut, 20 valeurs entre 2 et 100

        prices = []
        std_devs = []

        for steps in nb_steps_list:
            # print(steps)
            pricer = LSM_method(self.option)
            brownian = Brownian(self.period, steps, nb_paths, seed)
            price, std_error, _ = pricer.LSM(brownian, self.market, method='vector')

            prices.append(price)
            std_devs.append(std_error)

        prices = np.array(prices)
        std_devs = np.array(std_devs)

        # Calcul des intervalles de confiance à 95% (± 2 écart-types)
        lower_bound = prices - 2 * std_devs
        upper_bound = prices + 2 * std_devs

        mean_price = np.mean(prices)

        # Création du graphique Plotly
        fig = go.Figure()

        # Ajouter les prix avec barres d'erreur
        fig.add_trace(
            go.Scatter(
                x=nb_steps_list,
                y=prices,
                error_y=dict(
                    type='data',
                    array=2 * std_devs,
                    visible=True
                ),
                mode='markers',
                marker=dict(size=10, color='blue'),
                name="Prix estimés"
            )
        )

        # Ajouter les intervalles de confiance comme zone remplie
        fig.add_trace(
            go.Scatter(
                x=list(nb_steps_list) + list(nb_steps_list)[::-1],
                y=list(upper_bound) + list(lower_bound)[::-1],
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.1)',
                line=dict(color='rgba(0, 0, 255, 0)'),
                hoverinfo='skip',
                showlegend=False
            )
        )

        # Ajouter la ligne du prix moyen
        fig.add_trace(
            go.Scatter(
                x=[nb_steps_list[0], nb_steps_list[-1]],
                y=[mean_price, mean_price],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name=f"Prix moyen: {mean_price:.4f} ± {np.mean(std_devs):.4f}"
            )
        )

        # Mise en page
        fig.update_layout(
            title="Influence du nombre de pas sur le prix (± 2 écart-types)",
            xaxis_title="Nombre de pas (steps)",
            yaxis_title="Prix de l'option",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def comparer_seeds(self, nb_seeds=100, nb_paths=10000, nb_steps=20):
        """
        Compare les résultats pour différentes valeurs de seed.
        
        Args:
            nb_seeds: Nombre de seeds à tester
            nb_paths: Nombre de chemins pour la simulation
            nb_steps: Nombre de pas pour la simulation
            
        Returns:
            fig: Figure plotly
        """
        seeds = np.arange(1, nb_seeds + 1)
        prices = []
        std_devs = []
        
        for seed in seeds:
            pricer = LSM_method(self.option)
            brownian = Brownian(self.period, nb_steps, nb_paths, seed)
            price, std_error, _ = pricer.LSM(brownian, self.market, method='vector')
            
            prices.append(price)
            std_devs.append(std_error)
        
        prices = np.array(prices)
        std_devs = np.array(std_devs)
        
        # Calcul des intervalles de confiance à 95% (± 2 écart-types)
        lower_bound = prices - 2 * std_devs
        upper_bound = prices + 2 * std_devs
        
        mean_price = np.mean(prices)
        
        # Création du graphique Plotly
        fig = go.Figure()
        
        # Ajouter les prix avec barres d'erreur
        fig.add_trace(
            go.Scatter(
                x=seeds,
                y=prices,
                error_y=dict(
                    type='data',
                    array=2 * std_devs,
                    visible=True
                ),
                mode='markers',
                marker=dict(size=10, color='blue'),
                name="Prix estimés"
            )
        )
        
        # Ajouter les intervalles de confiance comme zone remplie
        fig.add_trace(
            go.Scatter(
                x=list(seeds) + list(seeds)[::-1],
                y=list(upper_bound) + list(lower_bound)[::-1],
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.1)',
                line=dict(color='rgba(0, 0, 255, 0)'),
                hoverinfo='skip',
                showlegend=False
            )
        )
        
        # Ajouter la ligne du prix moyen
        fig.add_trace(
            go.Scatter(
                x=[seeds[0], seeds[-1]],
                y=[mean_price, mean_price],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name=f"Prix moyen: {mean_price:.4f} ± {np.mean(std_devs):.4f}"
            )
        )
        
        # Mise en page
        fig.update_layout(
            title="Influence de la seed sur le prix (± 2 écart-types)",
            xaxis_title="Valeur de la seed",
            yaxis_title="Prix de l'option",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def comparer_polynomes(self, reference_price=None, poly_degrees=None, poly_types=None, 
                         nb_paths=10000, nb_steps=20):
        """
        Compare les résultats pour différents types et degrés de polynômes.
        
        Args:
            reference_price: Prix de référence (ex: price_BS ou price_tree)
            poly_degrees: Liste des degrés polynomiaux à tester
            poly_types: Liste des types de polynômes à tester
            nb_paths: Nombre de chemins pour la simulation
            nb_steps: Nombre de pas pour la simulation
            
        Returns:
            fig: Figure plotly
            df: DataFrame avec les résultats
        """
        if poly_degrees is None:
            poly_degrees = [2, 3, 4, 5, 6, 7]
        
        if poly_types is None:
            poly_types = ["Polynomial", "Laguerre", "Hermite", "Linear", 
                        "Logarithmic", "Exponential"]
        
        results = {}
        
        # Test des degrés polynomiaux (pour le type "polynomial")
        for degree in poly_degrees:
            brownian = Brownian(self.period, nb_steps, nb_paths, 1)
            pricer = LSM_method(self.option)
            
            start_time = time.time()
            price, std_error, _ = pricer.LSM(brownian, self.market, method='vector', 
                                          antithetic=False, poly_degree=degree, 
                                          model_type="Polynomial")
            end_time = time.time()
            execution_time = end_time - start_time
            
            key = f"Polynomial (deg={degree})"
            results[key] = {
                'price': price, 
                'std_error': std_error,
                'time': execution_time
            }
            
        # Test des types de modèles (avec degré polynomial fixé à 2)
        for poly_type in poly_types:
            if poly_type == "Polynomial":
                continue  # Déjà testé ci-dessus avec différents degrés
                
            brownian = Brownian(self.period, nb_steps, nb_paths, 1)
            pricer = LSM_method(self.option)
            
            start_time = time.time()
            price, std_error, _ = pricer.LSM(brownian, self.market, method='vector', 
                                          antithetic=False, poly_degree=2, 
                                          model_type=poly_type)
            end_time = time.time()
            execution_time = end_time - start_time
            
            results[poly_type] = {
                'price': price, 
                'std_error': std_error,
                'time': execution_time
            }
        
        # Création d'un DataFrame pour visualisation
        df_results = pd.DataFrame({
            'Model': list(results.keys()),
            'Price': [results[k]['price'] for k in results],
            'Std Error': [results[k]['std_error'] for k in results],
            'Time (s)': [results[k]['time'] for k in results]
        })
        
        # Création de la figure plotly avec deux sous-tracés
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "Comparaison des prix par type/degré de modèle (± 2 écart-types)",
            "Temps d'exécution par type/degré de modèle"
        ))
        
        models = df_results['Model']
        prices = df_results['Price']
        errors = df_results['Std Error']
        times = df_results['Time (s)']
        
        # Graphique des prix avec barres d'erreur
        for i, (model, price, error) in enumerate(zip(models, prices, errors)):
            color = self.colors[i % len(self.colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=[i],
                    y=[price],
                    error_y=dict(
                        type='data',
                        array=[2 * error],
                        visible=True
                    ),
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=model
                ),
                row=1, col=1
            )
        
        # Si un prix de référence est fourni
        if reference_price is not None:
            fig.add_shape(
                type="line",
                x0=-0.5, 
                y0=reference_price, 
                x1=len(models)-0.5, 
                y1=reference_price,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                ),
                row=1, col=1
            )
            
            fig.add_annotation(
                x=len(models)-1,
                y=reference_price,
                text=f"Prix de référence: {reference_price:.4f}",
                showarrow=False,
                yshift=10,
                row=1, col=1
            )
        else:
            fig.add_shape(
                type="line",
                x0=-0.5, 
                y0=np.mean(prices), 
                x1=len(models)-0.5, 
                y1=np.mean(prices),
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                ),
                row=1, col=1
            )
            
            fig.add_annotation(
                x=len(models)-1,
                y=np.mean(prices),
                text=f"Prix moyen: {np.mean(prices):.4f}",
                showarrow=False,
                yshift=10,
                row=1, col=1
            )
        
        # Graphique des temps d'exécution
        fig.add_trace(
            go.Bar(
                x=list(range(len(models))),
                y=times,
                marker_color=[self.colors[i % len(self.colors)] for i in range(len(models))],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Mise à jour des axes
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(len(models))),
            ticktext=models,
            tickangle=45,
            row=1, col=1
        )
        
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(len(models))),
            ticktext=models,
            tickangle=45,
            row=1, col=2
        )
        
        fig.update_yaxes(title_text="Prix de l'option", row=1, col=1)
        fig.update_yaxes(title_text="Temps d'exécution (s)", row=1, col=2)
        
        # Mise en page finale
        fig.update_layout(
            height=600,
            width=1400,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.50,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig, df_results
    
    def comparer_degres_par_type(self, poly_types=None, poly_degrees=None, 
                           nb_paths=10000, nb_steps=20):
        """
        Compare les résultats pour différents degrés pour chaque type de polynôme en utilisant Plotly.
        
        Args:
            poly_types: Liste des types de polynômes à tester
            poly_degrees: Liste des degrés polynomiaux à tester
            nb_paths: Nombre de chemins pour la simulation
            nb_steps: Nombre de pas pour la simulation
            
        Returns:
            fig: Figure Plotly
            df: DataFrame avec les résultats
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import time
        
        if poly_types is None:
            poly_types = ["Polynomial", "Laguerre", "Hermite"]
        
        if poly_degrees is None:
            poly_degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        results = {}
        
        for poly_type in poly_types:
            for degree in poly_degrees:
                brownian = Brownian(self.period, nb_steps, nb_paths, 1)
                pricer = LSM_method(self.option)
                
                try:
                    start_time = time.time()
                    price, std_error, _ = pricer.LSM(brownian, self.market, method='vector', 
                                                antithetic=False, poly_degree=degree, 
                                                model_type=poly_type)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    key = f"{poly_type} (deg={degree})"
                    results[key] = {
                        'type': poly_type,
                        'degree': degree,
                        'price': price, 
                        'std_error': std_error,
                        'time': execution_time
                    }
                except Exception as e:
                    print(f"Erreur pour {poly_type} degré {degree}: {e}")
        
        # Création d'un DataFrame pour visualisation
        df_results = pd.DataFrame([results[k] for k in results])
        
        # Définition des couleurs (compatible avec la palette originale)
        colors = {
            "Polynomial": "#1f77b4",  # bleu
            "Laguerre": "#ff7f0e",    # orange
            "Hermite": "#2ca02c"      # vert
        }
        
        # Création du graphique Plotly avec sous-graphiques
        fig = make_subplots(rows=len(poly_types), cols=1, 
                        subplot_titles=[f'Résultats pour le type {poly_type}' for poly_type in poly_types],
                        vertical_spacing=0.1)
        
        for i, poly_type in enumerate(poly_types):
            type_df = df_results[df_results['type'] == poly_type]
            
            if type_df.empty:
                continue
                
            # Tracer la ligne avec les prix
            fig.add_trace(
                go.Scatter(
                    x=type_df['degree'],
                    y=type_df['price'],
                    mode='lines+markers',
                    name=poly_type,
                    line=dict(color=colors.get(poly_type, f'rgba({(i*50)%255}, {(i*100)%255}, {(i*150)%255}, 1)')),
                    marker=dict(size=10)
                ),
                row=i+1, col=1
            )
            
            # Ajouter les barres d'erreur (2 × écart-type)
            fig.add_trace(
                go.Scatter(
                    x=type_df['degree'],
                    y=type_df['price'] + 2*type_df['std_error'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ),
                row=i+1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=type_df['degree'],
                    y=type_df['price'] - 2*type_df['std_error'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor=f'rgba({(i*50)%255}, {(i*100)%255}, {(i*150)%255}, 0.3)',
                    fill='tonexty',
                    name=f'{poly_type} Intervalle de confiance',
                    showlegend=False
                ),
                row=i+1, col=1
            )
        
        # Mise en forme du graphique
        fig.update_layout(
            height=300*len(poly_types),
            width=900,
            title_text="Comparaison des prix d'options par type et degré polynomial",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Configuration des axes
        for i in range(len(poly_types)):
            fig.update_xaxes(title_text="Degré polynomial", row=i+1, col=1, gridcolor='lightgray', gridwidth=0.5)
            fig.update_yaxes(title_text="Prix de l'option", row=i+1, col=1, gridcolor='lightgray', gridwidth=0.5)
        
        return fig, df_results