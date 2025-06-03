import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# ****************************************** LOSS FUNCTIONS ****************************************

# on a n echantillons ou samples, chaque echantillon va occasionner sa loss, donc la loss globale sera la moyenne de loss de tous les 
# samples, d'ou l'usage de np.mean(). Un sample sera un vecteur contenant le meme nbre de valeurs que de classes de sortie.
class Loss(ABC):
    
    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass
    
    def calculate_final_loss(self, output: np.ndarray, y: np.ndarray):
        sample_losses = self.compute_loss(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss # permet de faire la moyenne de tous les echantillons du dataset quelle que soit la fonction d'activation

# la consigne demande que la Binary soit utilisee car on a que deux resultats possibles en sortie : M ou B
class Loss_BinaryCrossEntropy(Loss):

    # la loss stricte nous dit "ou on est"
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)) # on sort le 1/m de la formule initiale
        return loss

    # le gradient de la loss nous indique "ou il faut aller"
    def compute_loss_gradient(self, y_pred: np.ndarray, y_true: np.ndarray): # le gradient de la loss => permettra de savoir comment ajuster les poids du modele
        y_true = np.array(y_true, dtype=np.float64)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        
        m = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        grad_loss = (y_pred_clipped - y_true) / m
        return grad_loss # on retourne le gradient de loss, par rapport aux predictions et non par rapport aux poids => a faire dans la BP


class Loss_CategoricalCrossEntropy(Loss):
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Si y_true est one-hot, convertir en indices
        if len(y_true.shape) == 2:
            y_true_indices = np.argmax(y_true, axis=1)
        else:
            y_true_indices = y_true
        
        correct_confidences = y_pred_clipped[range(len(y_pred)), y_true_indices]
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def compute_loss_gradient(self, y_pred: np.ndarray, y_true: np.ndarray):
        m = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            y_true_onehot = np.zeros_like(y_pred_clipped)
            y_true_onehot[np.arange(m), y_true.astype(int)] = 1
            y_true = y_true_onehot
        grad = (y_pred_clipped - y_true) / m
        return grad