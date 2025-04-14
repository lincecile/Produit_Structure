#%% Imports
from __future__ import annotations

import numpy as np

from Classes_Both.module_marche import DonneeMarche
from Classes_Both.module_option import Option
from Classes_Both.module_enums import ConventionBaseCalendaire, TypeBarriere, DirectionBarriere

#%%Constantes

somme_proba = 1
min_payoff = 0
prix_min_sj = 0 #on part du principe que le prix du sous-jacent ne peut être négatif (ce qui dans le cas d'un actif purement financier sera à priori toujours vrai)
epsilon = 1e-15 #notre seuil de pruning par défaut

#%% Classes

class Arbre : 
    def __init__(self, nb_pas : int, donnee_marche : DonneeMarche, option : Option,
                 convention_base_calendaire : ConventionBaseCalendaire = ConventionBaseCalendaire._365.value,
                 parametre_alpha : float = 3, pruning : bool = True, epsilon : float = epsilon) -> None:
        """Initialisation de la classe

        Args:
            nb_pas (float): le nombre de pas dans notre modèle
            donnee_marche (DonneeMarche): Classe utilisée pour représenter les données de marché.
            option (Option): Classe utilisée pour représenter une option et ses paramètres.
            convention_base_calendaire (ConventionBaseCalendaire, optional): la base calendaire que nous utiliserons pour nos calculs. Défaut ConventionBaseCalendaire._365.value.
            pruning (bool, optional): si l'on va ou non faire du "pruning". Défaut True.
        """
        self.nb_pas = nb_pas
        self.donnee_marche = donnee_marche
        self.option = option
        self.convention_base_calendaire = convention_base_calendaire
        self.parametre_alpha = parametre_alpha
        self.pruning = pruning
        self.epsilon = epsilon
        self.delta_t = self.__calcul_delta_t()
        self.facteur_capitalisation = self.__calcul_facteur_capitalisation()
        self.facteur_actualisation = self.__calcul_facteur_actualisation()
        self.position_div = self.__calcul_position_div()
        self.alpha = self.__calcul_alpha()
        self.racine = None
        self.prix_option = None
           
    def get_temps (self) -> float : 
        """Renvoie le temps à maturité exprimé en nombre d'année .

        Returns:
            float: temps à maturité en nombre d'année
        """
        return (self.option.maturite - self.option.date_pricing).days/self.convention_base_calendaire
    
    def __calcul_delta_t (self) -> float : 
        """Permet de calculer l'intervalle de temps de référence qui sera utilisée dans notre modèle.

        Returns:
            float: l'intervalle de temps delta_t
        """
        return self.get_temps() / self.nb_pas
    
    def __calcul_facteur_capitalisation(self) -> float : 
        """Permet de calculer le facteur de capitalisation que nous utiliserons par la suite

        Returns:
            float: un facteur de capitalisation à appliquer à chaque dt.
        """
        return np.exp(self.donnee_marche.taux_interet * self.delta_t)
    
    def __calcul_facteur_actualisation(self) -> float :
        """Permet de calculer le facteur d'actualisation que nous utiliserons par la suite


        Returns:
            float: un facteur d'actualisation à appliquer à chaque dt.
        """
        return np.exp(-self.donnee_marche.taux_actualisation * self.delta_t)
 
    def __calcul_alpha (self) -> float : 
        """Fonction nous permettant de calculer alpha, que nous utiliserons dans l'arbre.

        Returns:
            float: Nous renvoie le coefficient alpha
        """
        alpha = np.exp(self.donnee_marche.volatilite * np.sqrt(self.parametre_alpha) * np.sqrt(self.delta_t))
        
        return alpha 
    
    def __calcul_position_div (self) -> float : 
        """Nous permet de calculer la position du dividende dans l'arbre

        Returns:
            float: nous renvoie la position d'ex-date du div, exprimé en nombre de pas dans l'arbre.
        """
        nb_jour_detachement = (self.donnee_marche.dividende_ex_date - self.option.date_pricing).days
        position_div = nb_jour_detachement / self.convention_base_calendaire / self.delta_t
        
        return position_div
    
    def __planter_arbre(self) -> None : 
        """Procédure nous permettant de construire notre arbre
        """        
        def __creer_prochain_block_haut(actuel_centre : Noeud, prochain_noeud : Noeud) -> None : 
            """Procédure nous permettant de construire un bloc complet vers le haut à partir d'un noeud de référence et d'un noeud futur

            Args:
                actuel_centre (Noeud): notre noeud de référence
                prochain_noeud (Noeud): le noeud autour duquel nous allons créer le bloc
            """
            temp_centre = actuel_centre
            temp_futur_centre = prochain_noeud
            
            #Nous iterrons en partant du tronc et en nous dirigeant vers l'extrêmité haute d'une colonne afin de créer des noeuds sur la colonne suivante
            while not temp_centre.haut is None : 
                temp_centre = temp_centre.haut
                temp_centre._creer_prochain_block(temp_futur_centre)
                temp_futur_centre = temp_futur_centre.haut
                
        def __creer_prochain_block_bas(actuel_centre : Noeud, prochain_noeud : Noeud) -> None : 
            """Procédure nous permettant de construire un bloc complet vers le bas à partir d'un noeud de référence et d'un noeud futur

            Args:
                actuel_centre (Noeud): notre noeud de référence
                prochain_noeud (Noeud): le noeud autour duquel nous allons créer le bloc
            """            
            temp_centre = actuel_centre
            temp_futur_centre = prochain_noeud
            
            #Nous iterrons en partant du tronc et en nous dirigeant vers l'extrêmité basse d'une colonne afin de créer des noeuds sur la colonne suivante
            while not temp_centre.bas is None : 
                temp_centre = temp_centre.bas
                temp_centre._creer_prochain_block(temp_futur_centre)
                temp_futur_centre = temp_futur_centre.bas
                
        def __creer_nouvelle_col(self, actuel_centre : Noeud) -> Noeud :
            """Procédure nous permettant de créer entièrement une colonne de notre arbre.

            Args:
                actuel_centre (Noeud): le noeud sur le tronc actuel, que nous prenons en référence et à partir duquel nous créerons la colonne suivante.

            Returns:
                Noeud: nous renvoyons le futur noeud sur le centre afin de faire itérer cette fonction dessus
            """
            prochain_noeud = Noeud(actuel_centre._calcul_forward(), self, actuel_centre.position_arbre + 1)
            
            actuel_centre._creer_prochain_block(prochain_noeud)
            __creer_prochain_block_haut(actuel_centre, prochain_noeud)
            __creer_prochain_block_bas(actuel_centre, prochain_noeud)
            
            return prochain_noeud
        
        #Nous créons la racine de notre arbre ici, ne pouvant le faire au niveau de __init__ afin d'éviter un import récursif
        self.racine = Noeud(prix_sj = self.donnee_marche.prix_spot, arbre = self, position_arbre=0)
        
        #Notre première référence est la racine
        actuel_centre = self.racine
        
        #Nous créons ici le premier bloc. Nous itérerons ensuite sur autant de pas que nécéssaire afin de créer les colonnes suivantes.
        for pas in range(self.nb_pas) :
            actuel_centre = __creer_nouvelle_col(self, actuel_centre)
            
    def pricer_arbre(self) -> None : 
        """Fonction qui nous permettra de construire l'arbre puis de le valoriser pour enfin donner la valeur à l'attribut "prix_option".
        """
        self.__planter_arbre()
        
        self.racine._calcul_valeur_intrinseque()
        self.prix_option = self.racine.valeur_intrinseque

class Noeud:
    def __init__(self, prix_sj : float, arbre : Arbre, position_arbre : int) -> None:
        """Initialisation de la classe

        Args:
            prix_sj (float): le prix du sous-jacent de ce noeud
            arbre (Arbre): l'arbre auquel est rattaché notre noeud
            position_arbre (int): decrit la position du noeud dans l'arbre sur l'axe horizontal
        """
        self.epsilon = arbre.epsilon
        
        self.prix_sj = prix_sj
        self.arbre = arbre
        self.position_arbre = position_arbre 
        
        self.bas = None
        self.haut = None
        self.precedent_centre = None
        self.futur_bas = None
        self.futur_centre = None
        self.futur_haut = None 
        self.p_bas = None
        self.p_mid = None
        self.p_haut = None
        self.p_cumule = 1 if self.position_arbre == 0 else 0
        
        self.valeur_intrinseque = None
        
    def _calcul_forward(self) -> float : 
        """Permet de calculer la valeur du prix forward sur dt suivant
        
        Returns:
            float : prix forward
        """
        if self.position_arbre < self.arbre.position_div and self.position_arbre + 1 > self.arbre.position_div : 
            div = self.arbre.donnee_marche.dividende_montant
        else : 
            div = 0
            
        return self.prix_sj * self.arbre.facteur_capitalisation - div
    
    def __calcul_variance(self) -> float : 
        """Nous permet de calculer la variance

        Returns:
            float: variance
        """
        return (self.prix_sj ** 2) * np.exp(2 * self.arbre.donnee_marche.taux_interet * self.arbre.delta_t) * (np.exp((self.arbre.donnee_marche.volatilite ** 2) * self.arbre.delta_t) - 1)
    
    def __calcul_proba(self) -> None :
        """Nous permet de calculer les probabilités haut, centre, bas.
        """
        fw = self._calcul_forward()
       
        p_bas = ((self.futur_centre.prix_sj ** (-2)) * (self.__calcul_variance() + fw ** 2)
                      - 1 - (self.arbre.alpha + 1) * ((self.futur_centre.prix_sj ** (-1)) * fw - 1)) / ((1 - self.arbre.alpha) * (self.arbre.alpha ** (-2) - 1))
                     
        p_haut = (((1 / self.futur_centre.prix_sj * fw - 1) - (1 / self.arbre.alpha - 1) * p_bas) /
                       (self.arbre.alpha - 1))
        
        p_mid = 1 - p_haut - p_bas
        
        if not (p_bas > 0 and p_haut > 0 and p_mid > 0) :
            raise ValueError("Probabilité négative")
        
        if not np.isclose(p_bas + p_haut + p_mid, somme_proba, atol=1e-2) : 
            print(f"p_bas : {p_bas}, p_haut : {p_haut}, p_mid : {p_mid}")
            raise ValueError(f"La somme des probabilités doit être égale à {somme_proba}")
        else :             
            self.p_bas = p_bas
            self.p_haut = p_haut
            self.p_mid = p_mid      
            
    def __test_noeud_proche(self, forward : float) -> bool : 
        """Cette fonction nous permet de tester si le noeud est compris entre un prix d'un noeud haut ou d'un noeud bas.

        Args:
            forward (float): le prix forward de notre noeud que l'on aura calculé préalablement.

        Returns:
            bool: passage du test ou non 
        """
        condition_1 = (self.prix_sj * (1 + 1/self.arbre.alpha) / 2 <= forward)
        condition_2 = (forward <= self.prix_sj * (1 + self.arbre.alpha) / 2)
        if condition_1 and condition_2: 
            return True
        else : 
            return False
        
    def bas_suivant(self) -> Noeud : 
        """Nous permet de créer le noeud bas suivant si il n'existe pas déjà.

        Returns:
            Noeud: le noeud bas
        """
        if self.bas == None : 
            self.bas = Noeud(self.prix_sj / self.arbre.alpha, self.arbre, self.position_arbre)
            self.bas.haut = self
        return self.bas
            
    def haut_suivant(self) -> Noeud : 
        """Nous permet de créer le noeud haut suivant si il n'existe pas déjà.

        Returns:
            Noeud: le noeud haut
        """
        if self.haut == None : 
            self.haut = Noeud(self.prix_sj * self.arbre.alpha, self.arbre, self.position_arbre)
            self.haut.bas = self   
        return self.haut  
                
    def trouve_centre(self, prochain_noeud : Noeud) -> Noeud : 
        """Fonction nous permettant de retrouver le prochain noeud centre.

        Args:
            prochain_noeud (Noeud): noeud candidat

        Returns:
            Noeud: le centre de notre noeud de référence.
        """
        fw = self._calcul_forward()
        
        if prochain_noeud.__test_noeud_proche(fw) : 
            prochain_noeud = prochain_noeud
            
        elif fw > prochain_noeud.prix_sj : 
            while not prochain_noeud.__test_noeud_proche(fw) :
                prochain_noeud = prochain_noeud.haut_suivant()
        
        else : 
            while not prochain_noeud.__test_noeud_proche(fw) : 
                prochain_noeud = prochain_noeud.bas_suivant()
            
        return prochain_noeud

    def _creer_prochain_block(self, prochain_noeud : Noeud) -> None :
        """Nous permet de créer un bloc de noeud complet.

        Args:
            prochain_noeud (Noeud): _description_
        """
        self.futur_centre = self.trouve_centre(prochain_noeud=prochain_noeud)
        self.__calcul_proba()
        
        self.futur_centre.p_cumule += self.p_cumule * self.p_mid
        self.futur_centre.precedent_centre = self
        
        if self.arbre.pruning : 
            if  self.haut == None :
                if self.p_cumule * self.p_haut >= self.epsilon :  
                    self.futur_haut = self.futur_centre.haut_suivant()
                    self.futur_haut.p_cumule +=  self.p_cumule *  self.p_haut           
                else : 
                    # self.p_mid += self.p_haut
                    self.p_haut = 0
            elif not self.haut == None : 
                self.futur_haut = self.futur_centre.haut_suivant()
                self.futur_haut.p_cumule += self.p_cumule * self.p_haut
                
            if self.bas == None : 
                if self.p_cumule * self.p_bas >= self.epsilon : 
                    self.futur_bas = self.futur_centre.bas_suivant()
                    self.futur_bas.p_cumule += self.p_cumule * self.p_bas
                else : 
                    # self.p_mid += self.p_bas
                    self.p_bas = 0
            elif not self.bas == None : 
                self.futur_bas = self.futur_centre.bas_suivant()
                self.futur_bas.p_cumule += self.p_cumule * self.p_bas
              
        if not self.arbre.pruning :
            self.futur_haut = self.futur_centre.haut_suivant()
            self.futur_haut.p_cumule += self.p_cumule * self.p_haut
            self.futur_bas = self.futur_centre.bas_suivant()
            self.futur_bas.p_cumule += self.p_cumule * self.p_bas

    def __calcul_payoff(self) -> float : 
        """Calcul du payoff selon le type de contrat

        Returns:
            float: le payoff
        """
        option = self.arbre.option
        
        def call_put_payoff() : 
            return self.prix_sj - option.prix_exercice if option.call else option.prix_exercice - self.prix_sj
                    
        if option.barriere is not None and option.barriere.type_barriere is TypeBarriere.knock_in : 
            if ((option.barriere.direction_barriere is DirectionBarriere.up and self.prix_sj >= option.barriere.niveau_barriere) 
                or (option.barriere.direction_barriere is DirectionBarriere.down and self.prix_sj <= option.barriere.niveau_barriere)) :
                    payoff = max(min_payoff, call_put_payoff())
            else : 
                payoff = min_payoff
    
        elif option.barriere is not None and option.barriere.type_barriere is TypeBarriere.knock_out : 
            if ((option.barriere.direction_barriere is DirectionBarriere.up and self.prix_sj >= option.barriere.niveau_barriere) 
                or (option.barriere.direction_barriere is DirectionBarriere.down and self.prix_sj <= option.barriere.niveau_barriere)):
                payoff = min_payoff
            else:
                payoff = max(min_payoff, call_put_payoff())

        else : 
            payoff = max(call_put_payoff(), min_payoff)
            
        return payoff

    def _calcul_valeur_intrinseque(self) -> None : 
        """Nous permet de calculer la valeur intrinseque du noeud, en prenant compte du type d'option considéré
        """
        if self.futur_centre is None : 
            self.valeur_intrinseque = self.__calcul_payoff()
        
        elif self.valeur_intrinseque == None : 
            
            for futur_noeud in ["futur_haut", "futur_centre", "futur_bas"] :
                if getattr(self, futur_noeud) is None :
                    setattr(self, futur_noeud, Noeud(0, self.arbre, self.position_arbre+1))
                    noeud = getattr(self, futur_noeud)
                    noeud.valeur_intrinseque = 0
                else : 
                    noeud = getattr(self, futur_noeud)
                    if getattr(noeud, "valeur_intrinseque") is None  :
                        noeud._calcul_valeur_intrinseque()
                    
            vecteur_proba = np.array([self.p_haut, self.p_mid, self.p_bas]) #vecteur composé des probabilités des noeuds futurs du noeud actuel
            vecteur_prix = np.array([self.futur_haut.valeur_intrinseque, self.futur_centre.valeur_intrinseque, self.futur_bas.valeur_intrinseque]) #
            valeur_intrinseque = self.arbre.facteur_actualisation * vecteur_prix.dot(vecteur_proba) #ici, produit scalaire des prix par leurs probabilités
            
            if self.arbre.option.americaine : 
                payoff_dt = self.__calcul_payoff()
                valeur_intrinseque = max(payoff_dt, valeur_intrinseque)
                
            self.valeur_intrinseque = valeur_intrinseque