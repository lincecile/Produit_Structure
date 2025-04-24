#%% Imports

import streamlit as st
import datetime as dt
from datetime import timedelta
import numpy as np 
import time
import sys
import os
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
sys.setrecursionlimit(1000000000)

from src.options.Pricing_option.Classes_Both.module_enums import TypeBarriere, DirectionBarriere, ConventionBaseCalendaire, MethodeCalcul, RegType, SensOption, StratOption
from src.options.Pricing_option.Classes_Both.module_marche import DonneeMarche
from src.options.Pricing_option.Classes_Both.module_option import Option
from src.options.Pricing_option.Classes_TrinomialTree.module_barriere import Barriere
from src.options.Pricing_option.Classes_TrinomialTree.module_arbre_noeud import Arbre
from src.options.Pricing_option.Classes_Both.module_pricing_analysis import StrikeComparison, VolComparison, RateComparison
from src.options.Pricing_option.Classes_Both.module_black_scholes import BlackAndScholes
from src.options.Pricing_option.Classes_TrinomialTree.module_grecques_empiriques import GrecquesEmpiriques

from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_brownian import Brownian
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_LSM import LSM_method
from src.options.Pricing_option.Classes_MonteCarlo_LSM.module_graph import LSMGraph

from src.options.Pricing_option.Classes_Both.derivatives import OptionDerivatives, OptionDerivativesParameters

from src.Strategies_optionnelles.Portfolio_options import OptionsPortfolio
from src.Strategies_optionnelles.Strategies_predefinies import OptionsStrategy

#%% Constantes

today= dt.date.today()

#%% Streamlit

st.set_page_config(layout="wide")


# Titre de l'application
st.title("LSM Monte Carlo")
st.markdown("""
<p style="font-size:20px">
    <strong><a href="https://www.linkedin.com/in/emmaalaoui/" target="_blank">ALAOUI Emma</a>,
    <strong><a href="https://www.linkedin.com/in/imen-khadir-b51498205/" target="_blank">KHADIR Imen</a>,
    <strong><a href="https://www.linkedin.com/in/c%C3%A9cile-lin-196b751b5/" target="_blank">LIN Cécile</a>,
    <strong><a href="https://www.linkedin.com/in/mat%C3%A9o-molinaro/" target="_blank">MOLINARO Matéo</a>,
    <strong><a href="https://www.linkedin.com/in/enzomontariol/" target="_blank">MONTARIOL Enzo</a>
</p>
""", unsafe_allow_html=True)
# tab_LSM, tab_trinomial_tree = st.tabs(["LSM Monte Carlo","Trinomial Tree"])


###########################################################################
###################### Onglet 1 : Inputs Utilisateur ######################
########################################################################### 

tab_option_strat, tab1, tab2, tab3, tab4, tabcomparaison = st.tabs(["Strat Option", "Pricing", "Plus d'options", "Graphique : Brownien et Sous-Jacent", "Greeks", "Analyse - Comparaison"])


with tab1 :
    
    activer_pricing = st.button('Pricing')

    col11, col12, col13 = st.columns(3)
    
    with col11 : 
        date_pricing = st.date_input("Entrez une date de pricing :", value=today)
        
    with col13 : 
        nb_pas = st.number_input("Entrez le nombre de pas utilisé pour le pricing :", 1, 5000, value=100, step=1)
        nb_chemin = st.number_input("Entrez le nombre de chemin utilisé pour le pricing :", 1, 5000000, value=100000, step=1)
        
    st.divider()

    #Données de marché

    st.subheader("Données de marché :")
    
    dividende_check = st.toggle("Dividende", value=False)

    col21, col22, col23 = st.columns(3)

    with col21 : 

        spot = st.number_input("Entrez le prix de départ du sous-jacent (en €):", format="%.2f",value=100.0, step=0.01)
        
    with col22:
        
        volatite = st.number_input("Entrez le niveau de volatilité (en %):", format="%.2f", value=20.0, step=1.00)/100
        
    with col23:
        risk_free_rate = st.number_input("Entrez le niveau de taux d'intérêt (en %):", format="%.2f", value=4.0, step=1.00)/100
    
    if dividende_check : 
        with col21 : 
            dividende_ex_date = st.date_input("Entrez la date de dividende :")
        with col22:
            dividende_montant = st.number_input("Entrez le montant du dividende (en €):", format="%.2f" ,value=0.0, step=1.00)
    else : 
        dividende_ex_date = today
        dividende_montant=0
    
    #Option
    
    st.divider()
    
    st.subheader("Caractéristique de l'option :")
    
    col31, col32, col33 = st.columns(3)
    
    with col31:
        maturite = st.date_input("Entrez une date de maturité :",value=today+ timedelta(days=10))
        
    with col33:
        barriere_check = st.checkbox("Option à barrière ? (uniquement arbre trinomial)", value=False)
        
    col41, col42, col43 = st.columns(3)
        
    with col41 :
        strike = st.number_input("Entrez le strike (en €):", format="%.2f",value=100.0, step=0.01)
    
    with col42:
        option_type = st.selectbox("Choisissez le type de l'option :", ['Call', 'Put'])
        
    with col43:
        option_exercice = st.selectbox("Choisissez le type de l'exercice :", ['Européenne','Américaine']) 
    
    st.divider()
    col412, col422, col432 = st.columns(3)

    with col412 :
        # strike = st.number_input("Entrez le strike (en €):", format="%.2f",value=100.0, step=0.01)
        sens_option = st.selectbox("Choisissez le sens de l'option ou de la stratégie:", [sens.value for sens in SensOption])
    
    with col422:
        option_type_strat = st.selectbox("Choisissez une stratégie prédéfinie :", [strat.value for strat in StratOption])
        params = {"americaine": st.checkbox("Option américaine", value=True)}
        
    
        
        if option_type_strat in ["Call Spread", "Put Spread", "Strangle", 'Collar']:
            params["strike1"] = st.number_input("Entrez le strike de la première option :", 0.0, format="%.2f", value=95.0, step=0.01)
            params["strike2"] = st.number_input("Entrez le strike de la seconde option :", 0.0, format="%.2f", value=105.0, step=0.01)
        
        if option_type_strat in ["Straddle", "Long Call", "Short Call", "Long Put", "Short Put"]:
            params["strike"] = st.number_input("Entrez le strike des options :", 0.0, format="%.2f", value=100.0, step=0.01)
        
        if option_type_strat in ["Butterfly"]:
            params["strike1"] = st.number_input("Entrez le strike de la première option :", 0.0, format="%.2f", value=90.0, step=0.01)
            params["strike2"] = st.number_input("Entrez le strike de la seconde option :", 0.0, format="%.2f", value=100.0, step=0.01)
            params["strike3"] = st.number_input("Entrez le strike de la troisième option :", 0.0, format="%.2f", value=110.0, step=0.01)
            params["is_call"] = st.checkbox("Utiliser des calls (décocher pour des puts)", value=True)
    
    with col432:
        params["quantity"] = st.number_input("Quantité:",0, value=1, step=1)
        indice = st.number_input("Indice de l'option à supprimer:",0, value=1, step=1)

    #Barrière
        
    if barriere_check:
        
        st.divider()
        st.subheader("Barrière :")
        
        col51, col52, col53 = st.columns(3)
        
        with col51 : 
            niveau_barriere = st.number_input("Entrez le niveau de la barrière (en €):", format="%.2f",value=spot*1.1, step=0.01)
        
        with col52 :
            type_barriere_select = st.selectbox("Choisissez le type de barrière :", [type.value for type in TypeBarriere])
            type_barriere = TypeBarriere(type_barriere_select)
        
        with col53 : 
            direction_barriere_select = st.selectbox("Choisissez le sens de la barrière :", [direction.value for direction in DirectionBarriere])
            direction_barriere = DirectionBarriere(direction_barriere_select)
    else: 
        niveau_barriere=0
        type_barriere=None
        direction_barriere=None
    
#Ici, on feed les objets

barriere = Barriere(niveau_barriere=niveau_barriere, type_barriere=type_barriere, direction_barriere=direction_barriere)
    
donnee_marche = DonneeMarche(date_pricing, spot, volatite, risk_free_rate, risk_free_rate, dividende_ex_date, dividende_montant)
option = Option(maturite, strike, barriere=barriere, 
                americaine=False if option_exercice == 'Européenne' else True, 
                call=True if option_type == "Call" else False,
                date_pricing=date_pricing)

bs_check = option.americaine==False and donnee_marche.dividende_montant == 0 and option.barriere.direction_barriere == None

###########################################################################
############# Onglet 2 : Inputs additionnels Utilisateur ##################
###########################################################################  
    
with tab2 : 
    st.subheader("Plus de paramètre modulable")
    col11, col2, col3 = st.columns(3) 

    with col11:
        # on garde le format float, pour garder la possibilité de mettre 365.25
        convention_base_calendaire = st.selectbox('Choisissez la base annuelle :', [nombre.value for nombre in ConventionBaseCalendaire])

    st.divider()

    st.subheader("Plus de paramètre modulable pour le pricing LSM")
    col11, col2, col3 = st.columns(3) 

    with col11:
        # on garde le format float, pour garder la possibilité de mettre 365.25
        calcul_method = st.selectbox('Choisissez la méthode de calcul :', [nombre.value for nombre in MethodeCalcul])
        regress_method = st.selectbox('Choisissez la méthode de regression :', [nombre.value for nombre in RegType])
        seed_choice = st.number_input('Choisissez la seed pour le pricing LSM:', 1, 50000, value=42, step=1)

    with col3:
        calcul_antithetic = st.toggle("Calcul Antithétique", value=False)
        antithetic_choice = False
        if calcul_antithetic : 
            antithetic_choice = True
            st.markdown("Calcul antithétique activé")

        if regress_method in ["Polynomial","Laguerre","Legendre","Chebyshev","Hermite"]:
            poly_degree = st.number_input("Entrez le degré du polynôme :", 1, 100, value=2, step=1)
            

    st.divider()

    st.subheader("Plus de paramètre modulable pour l'arbre trinomial")
    col11, col2, col3 = st.columns(3) 

    with col11:
        # on garde le format float, pour garder la possibilité de mettre 365.25
        parametre_alpha = st.number_input("Entrez le paramètre alpha pour l'arbre trinomial:", min_value=2.0,max_value=4.0, value=3.0, step=0.1)

    with col3:
        pruning = st.toggle("Elagage de l'arbre", value=True)
        if pruning : 
            epsilon_arbre = float("1e-" + str(st.number_input('Seuil de pruning (1e-)', min_value = 1, max_value=100, value = 15)))
            st.markdown(epsilon_arbre)
            arbre = Arbre(nb_pas=nb_pas, donnee_marche=donnee_marche, option=option, convention_base_calendaire=convention_base_calendaire, parametre_alpha=parametre_alpha, pruning=pruning, epsilon=epsilon_arbre)
        else :
            arbre = Arbre(nb_pas=nb_pas, donnee_marche=donnee_marche, option=option, convention_base_calendaire=convention_base_calendaire, parametre_alpha=parametre_alpha, pruning=pruning)
    st.divider()

# feed objet LSM
brownian = Brownian(time_to_maturity=(maturite-date_pricing).days / convention_base_calendaire, nb_step=nb_pas, nb_trajectoire=nb_chemin, seed=seed_choice)
pricer = LSM_method(option)

# portfolio = OptionsPortfolio(brownian,donnee_marche)

with tab1:

    if activer_pricing : 
        if  bs_check : 
            bns = BlackAndScholes(modele=arbre)
            pricing_bns = f"{round(bns.bs_pricer(),2)}€"
            st.divider()
            st.subheader('Pricing Black and Scholes : ')
            st.metric('', value = pricing_bns)

        st.divider()
        st.subheader('Pricing LSM : ')
        

        start = time.time()
        with st.spinner('''Valorisation de l'option en cours...''') : 

            price, std_error, intevalles = pricer.LSM(brownian, donnee_marche, 
                                                      method= 'vector' if calcul_method == 'Vectorielle' else 'scalar', 
                                                      antithetic = antithetic_choice,
                                                      poly_degree=poly_degree, model_type=regress_method)
        end = time.time()
        time_difference = round(end - start, 1)
        prix_option = f"{round(price, 2)}€"
        std_error = f"{round(std_error, 4)}€"
        intevalles = f"{round(intevalles[0], 4)}€ - {round(intevalles[1], 4)}€"
        
        col11_LSM, col2_LSM, col3_LSM = st.columns(3) 
        with col11_LSM:
            st.metric('''Valeur de l'option :''', value=prix_option, delta=None)
            st.metric('Temps de pricing (secondes) :', value=time_difference, delta=None)
        with col2_LSM:
            st.metric('''Ecart type du prix de l'option :''', value=std_error, delta=None)
        with col3_LSM:
            st.metric('''Intervalle de confiance :''', value=intevalles, delta=None)

        st.divider()
        st.subheader('Pricing avec Arbre : ')

        start = time.time()
        with st.spinner('''Valorisation de l'option en cours...''') : 
            arbre.pricer_arbre()
        end = time.time()
        time_difference = round(end - start, 1)
        prix_option = f"{round(arbre.prix_option, 2)}€"
        
        # arbre_st = arbre
        
        st.metric('''Valeur de l'option :''', value=prix_option, delta=None)
        st.metric('Temps de pricing (secondes) :', value=time_difference, delta=None)
   
    if st.button('Ajouter une option au portfeuille') :
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = OptionsPortfolio(brownian, donnee_marche)
        
        st.session_state.portfolio.add_option(option, 1*params["quantity"] if sens_option == 'Long' else -1*params["quantity"])  
    
    if st.button('Ajouter une stratégie prédéfinie au portfeuille') :
        strategy_name = option_type_strat.lower()
        
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = OptionsPortfolio(brownian, donnee_marche)

        strategy = OptionsStrategy(st.session_state.portfolio, donnee_marche, expiry_date=maturite)
        
        strategy.create_strategy(option_type_strat, params, 1 if sens_option == 'Long' else -1)  
        st.success(f"Stratégie {option_type_strat} créée avec succès !")
    
    if st.button('Récap du portfeuille') :

        if 'portfolio' not in st.session_state:
            st.error("Le portefeuille est vide")
        else:
            summary_folio = st.session_state.portfolio.get_portfolio_summary()
            st.dataframe(summary_folio)

    if st.button('Grecques du portfeuille') :

        if 'portfolio' not in st.session_state:
            st.error("Le portefeuille est vide")
        else:
            greeks_folio = st.session_state.portfolio.calculate_portfolio_greeks()
            st.dataframe(greeks_folio)

    if st.button('Détail du portfeuille') :

        if 'portfolio' not in st.session_state:
            st.error("Le portefeuille est vide")
        else:
            detail_folio = st.session_state.portfolio.get_portfolio_detail()
            st.dataframe(detail_folio)
    
    # Bouton pour supprimer une option du portefeuille
    if st.button("Supprimer une/des options"):
        if 'portfolio' in st.session_state:
            st.session_state.portfolio.remove_option_quantity(indice,params["quantity"])
            st.success("Supprimé")

    # Bouton pour vider le portefeuille
    if st.button("Vider le portefeuille"):
        if 'portfolio' in st.session_state:
            st.session_state.portfolio.clear_portfolio()
            st.success("Le portefeuille a été vidé")

    if st.button('Tracer le payoff du portefeuille'):
        if 'portfolio' in st.session_state and st.session_state.portfolio.options :
            fig = st.session_state.portfolio.plot_portfolio_payoff(show_individual=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Le portefeuille est vide")

    # Pour afficher le payoff d'une option spécifique
    if st.button('Tracer le payoff de l\'option 1'):
        if 'portfolio' in st.session_state and st.session_state.portfolio.options:
            fig = st.session_state.portfolio.plot_option_payoff(0)  # Option à l'indice 0
            st.plotly_chart(fig)
        else:
            st.error("Le portefeuille est vide")



# ###########################################################################
# ################## Onglet 3 : Graphique Display #####################
# ###########################################################################  

graph = LSMGraph(option=option, market=donnee_marche)

with tab3 : 
    st.subheader("Brownien - Sous Jacent")
    col1, col2, col3 = st.columns(3)

    with col1:
        graph_boutton = st.button('Obtenir les mouvements browniens')

    with col3:
        graph_boutton_spot = st.button('Obtenir les trajectoires de prix sous-jacent')

    if graph_boutton : 
        with st.spinner("Graphique en cours de réalisation..."):
            graph_b = graph.afficher_mouvements_browniens(brownian,nb_trajectoires=nb_chemin)
            st.plotly_chart(graph_b, use_container_width=True)
    if graph_boutton_spot : 
        with st.spinner("Graphique en cours de réalisation..."):
            trajectoire = pricer.Price(market=donnee_marche, brownian=brownian)
            graph_spot = graph.afficher_trajectoires_prix(trajectoire,brownian,nb_trajectoires=nb_chemin)
            st.plotly_chart(graph_spot, use_container_width=True)
        

# ###########################################################################
# ########################### Onglet 4 : Grecques ###########################
# ########################################################################### 

with tab4 : 
    
    if arbre.prix_option is not None : 

        st.subheader("Grecques empiriques via la méthode LSM: ")
        
        option_deriv = OptionDerivatives(option, donnee_marche, pricer)  
        
        with st.spinner('''Calcul des grecques en cours...''') :
            delta = round(option_deriv.delta(brownian),2)
            gamma = round(option_deriv.gamma(brownian),2)
            vega = round(option_deriv.vega(brownian)/100,2)
            vomma = round(option_deriv.vomma(brownian)/100,2)
            # theta = round(option_deriv.theta(brownian)/100,2)
            rho = round(option_deriv.rho(brownian)/100,2)
            
        
        col11, col12, col13, col14, col15 = st.columns(5)

        with col11 : 
            st.metric(label='Delta',value=delta, delta=None)
        with col12 : 
            st.metric(label='Gamma',value=gamma, delta=None)
        with col13 : 
            st.metric(label='Vega',value=vega, delta=None)
        with col14 : 
            st.metric(label='Vomma',value=vomma, delta=None)
        with col15 : 
            st.metric(label='Rho',value=rho, delta=None)
        
        st.divider()

        st.subheader("Grecques empiriques via l'arbre Trinomial: ")
                    
        grecques_empiriques = GrecquesEmpiriques(arbre, var_s=0.01, var_v=0.01, var_t=1, var_r=0.01)
                
        with st.spinner('''Calcul des grecques en cours...''') :
            delta = round(grecques_empiriques.approxime_delta(),2)
            gamma = round(grecques_empiriques.approxime_gamma(),2)
            vega = round(grecques_empiriques.approxime_vega(),2)
            theta = round(grecques_empiriques.approxime_theta(),2)
            rho = round(grecques_empiriques.approxime_rho(),2)
        
        col11, col12, col13, col14, col15 = st.columns(5)

        with col11 : 
            st.metric(label='Delta',value=delta, delta=None)
        with col12 : 
            st.metric(label='Gamma',value=gamma, delta=None)
        with col13 : 
            st.metric(label='Vega',value=vega, delta=None)
        with col14 : 
            st.metric(label='Theta',value=theta, delta=None)
        with col15 : 
            st.metric(label='Rho',value=rho, delta=None)
            
    else : 
        st.markdown("Veuillez valoriser l'option via son arbre avant de pouvoir accéder à ses grecques calculée via avec l'arbre trinomial et la méthode LSM.")

    
    if bs_check : 
        st.divider()
        st.subheader('Grecques Black and Scholes : ')
    
        bs = BlackAndScholes(arbre)
        
        bs_delta = round(bs.delta(),2)
        bs_gamma = round(bs.gamma(),2)
        bs_vega = round(bs.vega(),2)
        bs_theta = round(bs.theta(),2)
        bs_rho = round(bs.rho(),2)
        
        col21, col22, col23, col24, col25 = st.columns(5)
        
        with col21 : 
            st.metric(label='Delta',value=bs_delta, delta=None)
        with col22 : 
            st.metric(label='Gamma',value=bs_gamma, delta=None)
        with col23 : 
            st.metric(label='Vega',value=bs_vega, delta=None)
        with col24 : 
            st.metric(label='Theta',value=bs_theta, delta=None)
        with col25 : 
            st.metric(label='Rho',value=bs_rho, delta=None)
            
        
# ###########################################################################
# ####################### Onglet 4 : Comparaison ############################
# ###########################################################################  

with tabcomparaison: 
    tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14 = st.tabs([ "Comparaison Vectoriel vs Scalaire", 
                                                                                "Comparaison Seed", "Comparaison Nombre de pas", 
                                                                                "Comparaison temps d'execution - nombre de chemin", 
                                                                                "Comparaison Polynôme", "Comparaison Degré polynôme", 
                                                                                "Comparaison Strike","Comparaison Volatilité", "Comparaison Taux d'intérêt"])


    # ###########################################################################
    # #################### Onglet : Vectoriel vs Scalaire #######################
    # ########################################################################### 

    with tab6 :

        vectoriel_scalaire_comparison_button = st.button('''Lancer l'analyse comparative''')
        
        if vectoriel_scalaire_comparison_button: 
            with st.spinner("Graphique en cours de réalisation..."):
                now = time.time()
                graph_vs = graph.comparer_methodes()
                st.markdown(f'Résultats après {round(time.time()-now,2)} secondes.')
                st.plotly_chart(graph_vs, use_container_width=True)

    # ###########################################################################
    # #################### Onglet : Comparaison Seed ############################
    # ########################################################################### 

    with tab7 :
        seed_comparison_button = st.button('''Lancer l'analyse comparative selon la seed''')
        
        if seed_comparison_button: 
            with st.spinner("Graphique en cours de réalisation..."):
                now = time.time()
                graph_seed = graph.comparer_seeds()
                st.markdown(f'Résultats après {round(time.time()-now,2)} secondes.')
                st.plotly_chart(graph_seed, use_container_width=True)
            
    
    # ###########################################################################
    # #################### Onglet : Nombre de pas ###############################
    # ########################################################################### 

    with tab8 :
        nb_pas_comparison_button = st.button('''Lancer l'analyse pour le nombre de pas''')
        
        if nb_pas_comparison_button: 
            with st.spinner("Graphique en cours de réalisation..."):
                now = time.time()
                graph_nb_pas = graph.comparer_steps()
                st.markdown(f'Résultats après {round(time.time()-now,2)} secondes.')
                st.plotly_chart(graph_nb_pas, use_container_width=True)
    
    # ###########################################################################
    # ######################## Onglet : Nombre de chemin ########################
    # ########################################################################### 

    with tab9 :
        path_comparison_button = st.button('''Lancer l'analyse comparative selon le nombre de chemin''')
        
        if path_comparison_button: 
            with st.spinner("Graphique en cours de réalisation..."):
                now = time.time()
                graph_path = graph.comparer_convergence_paths()
                st.markdown(f'Résultats après {round(time.time()-now,2)} secondes.')
                st.plotly_chart(graph_path, use_container_width=True)
            
    # ###########################################################################
    # #################### Onglet : Comparaison Polynôme ########################
    # ########################################################################### 

    with tab10 :
        poly_comparison_button = st.button('''Lancer l'analyse comparative selon le polynôme''')
        
        if poly_comparison_button: 
            with st.spinner("Graphique en cours de réalisation..."):
                now = time.time()
                graph_poly, df_results = graph.comparer_polynomes()
                st.markdown(f'Résultats après {round(time.time()-now,2)} secondes.')
                st.plotly_chart(graph_poly, use_container_width=True)
                # st.dataframe(df_results)

    # ###########################################################################
    # #################### Onglet : Comparaison Degré ###########################
    # ########################################################################### 

    with tab11 :
        degree_comparison_button = st.button('''Lancer l'analyse comparative selon le degré des polynômes''')
        
        if degree_comparison_button:

            st.subheader('''Prix LSM selon le niveau de degré des polynômes''')

            with st.spinner("Graphique en cours de réalisation..."):
                now = time.time()
                graph_degree, df_results = graph.comparer_degres_par_type()
                st.markdown(f'Résultats après {round(time.time()-now,2)} secondes.')
                st.plotly_chart(graph_degree, use_container_width=True)
                # st.dataframe(df_results)


    # ###########################################################################
    # ##################### Onglet : Strike Comparaison #########################
    # ###########################################################################

    with tab12 : 
        max_cpu = st.number_input('''Veuillez choisir le nombre de coeur qui sera mis à contribution pour le multiprocessing (choisir 1 revient à ne pas en faire, monter trop haut peut induire de l'instabilité.):''',1,os.cpu_count(),4,1,key='strike')
        strike_comparison_button = st.button('''Lancer l'analyse comparative (l'opération prend environ 2min)''')
        
        if strike_comparison_button:
            now = time.time()
            # @st.cache_resource
            def call_strike_comparison() :
                step_list=[20]
                strike_list = np.arange(spot-10, spot+10, 0.5)
                return StrikeComparison(max_cpu, step_list, strike_list, donnee_marche, brownian, option)
            
            st.subheader('''Différence d'écart selon le niveau de strike''')

            strike_comparison = call_strike_comparison()

            st.markdown(f'Résultats après {round(time.time()-now,2)} secondes.')
            st.plotly_chart(strike_comparison.graph_strike_comparison())
            
            with st.expander(label='Données'): 
                st.markdown(f'''Pour une option {option_type} de type {option_exercice}: avec un prix de départ du sous jacent à {spot}, une volatilité à {volatite} et un taux d'intérêt à {risk_free_rate}, une date de pricing au {date_pricing} et une maturité au {maturite}, on obtient le tableau suivant en fonction du strike.''')
                st.dataframe(strike_comparison.results_df.sort_values(by='Strike',ascending=True))

    # ###########################################################################
    # ################## Onglet 7 : Volatilité Comparaison ######################
    # ###########################################################################

    with tab13 : 
        max_cpu = st.number_input('''Veuillez choisir le nombre de coeur qui sera mis à contribution pour le multiprocessing (choisir 1 revient à ne pas en faire, monter trop haut peut induire de l'instabilité.):''',1,os.cpu_count(),4,1,key='vol')
        vol_comparison_button = st.button('''Lancer l'analyse comparative (l'opération prend environ 1.5min)''')
        
        if vol_comparison_button :
            now = time.time()
            # @st.cache_resource
            def call_vol_comparison():
                step_list=[20]
                vol_list=np.arange(0.01,1,0.01)
                return VolComparison(max_cpu,step_list, vol_list, donnee_marche, brownian, option)
            
            vol_comparison = call_vol_comparison()
            
            st.markdown(f'Résultats après {round(time.time()-now,2)} secondes.')
            st.plotly_chart(vol_comparison.graph_vol())
            
            with st.expander(label='Données'): 
                st.markdown(f'''Pour une option {option_type} de type {option_exercice}: avec un prix de départ du sous jacent à {spot}, une strike à {strike} et un taux d'intérêt à {risk_free_rate}, une date de pricing au {date_pricing} et une maturité au {maturite}, on obtient le tableau suivant en fonction de la voltatilité.''')
                st.dataframe(vol_comparison.results_df.sort_values(by='Volatilité',ascending=True))
                
    # ###########################################################################
    # ################## Onglet 7 : Intérêt Comparaison ######################
    # ###########################################################################

    with tab14 : 
        max_cpu = st.number_input('''Veuillez choisir le nombre de coeur qui sera mis à contribution pour le multiprocessing (choisir 1 revient à ne pas en faire, monter trop haut peut induire de l'instabilité.):''',1,os.cpu_count(),4,1,key='rate')
        rate_comparison_button = st.button('''Lancer l'analyse comparative (l'opération prend environ 2 min)''')
        
        if rate_comparison_button :
            now = time.time()
            @st.cache_resource
            def call_rate_comparison():
                step_list=[20]
                rate_list=np.arange(-0.5,0.5,0.01)
                return RateComparison(max_cpu, step_list, rate_list, donnee_marche, brownian, option)
            
            rate_comparison = call_rate_comparison()
            
            st.markdown(f'Résultats après {round(time.time()-now,2)} secondes.')
            st.plotly_chart(rate_comparison.graph_rate())
            
            with st.expander(label='Données'): 
                st.markdown(f'''Pour une option {option_type} de type {option_exercice}: avec un prix de départ du sous jacent à {spot}, une strike à {strike} et une volatilité à {volatite}, une date de pricing au {date_pricing} et une maturité au {maturite}, on obtient le tableau suivant en fonction du taux d'intérêt.''')
                st.dataframe(rate_comparison.results_df.sort_values(by='Taux d\'intérêt',ascending=True))

