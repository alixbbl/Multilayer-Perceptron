import numpy as np
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

    def compute_loss(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        loss = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return loss

    def compute_gradient(self, y_pred, y_true): # gradient est la derivee de la fonction de perte
        m = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        grad = (-(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped))) / m
        return grad


class Loss_CategoricalCrossEntropy(Loss):
    
    def compute_loss(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # on clip les valeurs de 0 et 1 dont les logarithmes vont faire crasher les calculs
        # si y_true est une liste d'indices de classes comme [1, 3, 3, 0, 2, 1] avec 0, 1, 2, 3 les ID des classes
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # si y_true est une matrice issue d'un one-hot encoding avec juste des 0 et 1 pour Faux/Vrai
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_likelihoods = -np.log(correct_confidences)
        return negative_likelihoods

    def compute_gradient(self, y_pred, y_true):
        m = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        grad = (y_pred_clipped - y_true) / m
        return grad