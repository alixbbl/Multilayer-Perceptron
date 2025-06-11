import numpy as np

# **************************************** ACTIVATION FUNCTIONS **************************************

np.random.seed(0)

#pour les hidden layers
class Activation_reLU:
    def activate(self, inputs: np.ndarray):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def derivative(self, inputs):
        return (inputs > 0).astype(float)

class Activation_Sigmoid:
    def activate(self, inputs):
        self.inputs = inputs
        self.outputs = np.where(inputs >= 0, 
                               1 / (1 + np.exp(-inputs)),
                               np.exp(inputs) / (1 + np.exp(inputs)))
        return self.outputs
    
    def derivative(self, inputs):
        return self.outputs * (1 - self.outputs)

class Activation_SoftMax:
    def activate(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self.outputs = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        return self.outputs
    
    def derivative(self, inputs):
        return self.outputs * (1 - self.outputs)