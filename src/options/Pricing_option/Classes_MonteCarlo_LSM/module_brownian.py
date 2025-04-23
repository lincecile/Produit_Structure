import numpy as np
import scipy.stats as stats

class Brownian:
    def __init__(self, time_to_maturity: float, nb_step: int, nb_trajectoire: int, seed: int) -> None:
        self.nb_step = nb_step
        self.nb_trajectoire = nb_trajectoire
        self.step = time_to_maturity / self.nb_step
        self.seed = seed
        # self._generator : np.random.Generator = np.random.default_rng(self.seed)
        np.random.seed(self.seed)

    def Scalaire(self) -> np.ndarray:
        # Mouvement Brownien scalaire
        W = np.zeros(self.nb_step+1) 
        for i in range(1,self.nb_step+1): 
            uniform_samples = np.random.uniform(0, 1)
            W[i] = W[i-1]+stats.norm.ppf(uniform_samples) * np.sqrt(self.step)
        return W

    def Vecteur(self)-> np.ndarray:
        # Génération vectorielle des mouvements browniens
        uniform_samples = np.random.uniform(0, 1, (self.nb_trajectoire,self.nb_step)) 
        Z = stats.norm.ppf(uniform_samples)
        dW = np.sqrt(self.step) * Z
        W = np.zeros((self.nb_trajectoire, self.nb_step+1))
        W[:, 1:] = np.cumsum(dW, axis=1)
                
        return W


