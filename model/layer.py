import numpy as np
from abc import ABC, abstractmethod
from model.activation import Activation_SoftMax, Activation_Sigmoid, Activation_reLU

# ****************************************** DENSELAYER FUNCTIONS ****************************************

class DenseLayer:

    def __init__(self, n_inputs, n_neurons, activation='relu'):
        if activation == 'relu':
            self.activation = Activation_reLU()
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)  # He
        elif activation == 'Sigmoid':
            self.activation = Activation_Sigmoid()
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1. / n_inputs)  # Xavier
        elif activation == 'SoftMax':
            self.activation = Activation_SoftMax()
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1. / n_inputs)  # Xavier
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.outputs = self.activation.activate(self.z)  # attention au return
        return self.outputs # on retourne toujours vers la loss, les outputs actives et non pas tels que !

    # elle prend en argument le gradient de la loss par rapport aux outputs de la couche suivante, une fois la prop finie
    # dWeights et dBiases, ce sont les derives 
    def backward(self, grad_dOutputs: np.ndarray) -> np.ndarray:
        # Gradient de l'activation
        grad_activation = grad_dOutputs * self.activation.derivative(self.z)
        
        # Gradients des paramètres
        self.dWeights = np.dot(self.inputs.T, grad_activation)
        self.dBiases = np.sum(grad_activation, axis=0, keepdims=True)
        
        # Gradient pour la couche précédente
        self.dInputs = np.dot(grad_activation, self.weights.T)
        return self.dInputs

