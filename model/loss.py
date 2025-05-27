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
    def calculate_final_loss(self, output, y):
        sample_losses = self.compute_loss(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss # permet de faire la moyenne de tous les echantillons du dataset quelle que soit la fonction d'activation

# la consigne demande que la Binary soit utilisee car on a que deux resultats possibles en sortie : M ou B
class Loss_BinaryCrossEntropy(Loss):

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return loss

    def compute_gradient(self, y_pred, y_true):
        y_true = np.array(y_true, dtype=np.float64).reshape(-1, 1)
        m = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        grad = (y_pred_clipped - y_true) / m
        return grad


class Loss_CategoricalCrossEntropy(Loss):
    
    def compute_loss(self, y_pred: pd.Series, y_true: pd.Series):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Si y_true est one-hot => labls
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def compute_gradient(self, y_pred: pd.Series, y_true: pd.Series):
        m = y_true.shape[0]
        y_true = np.array(y_true, dtype=np.float64)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        grad = (y_pred_clipped - y_true) / m
        return grad