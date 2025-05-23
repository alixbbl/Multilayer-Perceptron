import numpy as np

# **************************************** ACTIVATION FUNCTIONS **************************************

np.random.seed(0)

#pour les hidden layers
class Activation_reLU:
    def activate(self, inputs: np.ndarray):
        self.outputs = np.maximum(0, inputs)

class Activation_Sigmoid:
    def activate(self, inputs):
        self.outputs = np.where(inputs >= 0, 1 / (1 + np.exp(-inputs)), np.exp(inputs) / (1 + np.exp(inputs)))

# pour la output layer => demande de la consigne
class Activation_SoftMax:
    def activate(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        self.outputs = probabilities
