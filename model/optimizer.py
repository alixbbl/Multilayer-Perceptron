# ****************************************** GRADIENT DESCENT *****************************************

# ajouter le suivi de la loss pour la partie graphique de la consigne
class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def step(self, weights, gradient):
        weights -= self.learning_rate * gradient
        return weights