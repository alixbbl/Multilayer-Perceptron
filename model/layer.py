import numpy as np
from abc import ABC, abstractmethod

# ****************************************** DENSELAYER FUNCTIONS ****************************************

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, loss): 
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # on devrait faire l'inverse, on evite d'utiliser la .T => essayer avec! 
        self.biases = np.zeros((1, n_neurons)) # on fait toujours un tableau du nombre de neurones, un biais / neurone.
        self.activation = None

    def forward(self, inputs: np.ndarray): # input peut etre de la data en entree de la input_layer ou les resultats de la cocuhe precedente
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases # cet output reste a activer !

    # backward() s'applique sur CHAQUE NEURONE
    # elle prend en argument le gradient de la loss par rapport aux outputs de la couche suivante, une fois la prop finie
    # dWeights et dBiases, ce sont les derives 
    def backward(self, grad_dOutputs: np.ndarray):
        self.dWeights = np.dot(self.inputs.T, grad_dOutputs) # calcule la derivee grace aux inputs connus et au gradient de la loss de la couche finale 
        self.dBiases = np.sum(grad_dOutputs, axis=0, keepdims=True) # le biais est une constante, il n'est pas influence par les input ! on le retrouve donc 
        self.dInputs = np.dot(grad_dOutputs, self.weights.T)