import numpy as np
from .activation import Activation_SoftMax, Activation_Sigmoid, Activation_reLU

# ****************************************** DENSELAYER FUNCTIONS ****************************************

class DenseLayer:

    def __init__(self, n_inputs, n_neurons, activation: str):
        self.n_inputs = n_inputs
        self.biases = np.zeros((1, n_neurons))

        if activation.lower() == "relu":
            self.activation = Activation_reLU()
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs) # He init
        elif activation == "Sigmoid":
            self.activation = Activation_Sigmoid()
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        elif activation == "SoftMax":
            self.activation = Activation_SoftMax()
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs) # Xavier Init
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, inputs: np.ndarray):

        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.outputs = self.activation.activate(self.z)
        return self.outputs

    def backward(self, grad_from_next: np.ndarray) -> np.ndarray:
        
        if isinstance(self.activation, Activation_SoftMax):
            grad_activation = grad_from_next
        else:
            grad_activation = grad_from_next * self.activation.derivative(self.z)
        
        self.dWeights = np.dot(self.inputs.T, grad_activation)
        self.dBiases = np.sum(grad_activation, axis=0, keepdims=True)
        
        self.dInputs = np.dot(grad_activation, self.weights.T)
        
        return self.dInputs
