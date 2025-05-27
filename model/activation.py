import numpy as np

# **************************************** ACTIVATION FUNCTIONS **************************************

np.random.seed(0)

#pour les hidden layers
class Activation_reLU:
    def activate(self, inputs: np.ndarray):
        self.outputs = np.maximum(0, inputs)
    
    def derivative(self, inputs):
        return (inputs > 0).astype(float)


class Activation_Sigmoid:
    def activate(self, inputs):
        self.inputs = inputs
        self.outputs = np.empty_like(inputs)
        positive = inputs >= 0
        negative = ~positive
        # Cas inputs >= 0
        self.outputs[positive] = 1 / (1 + np.exp(-inputs[positive]))
        # Cas inputs < 0
        exp_x = np.exp(inputs[negative])
        self.outputs[negative] = exp_x / (1 + exp_x)
    
    def derivative(self, inputs):
        return self.outputs * (1 - self.outputs)


class Activation_SoftMax:
    def activate(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        self.outputs = probabilities


#    def derivative(self, inputs):
#         # Utiliser les inputs (pre-activation) pour calculer sigmoid puis sa dérivée
#         sig = self.activate_without_storing(inputs)
#         return sig * (1 - sig)
    
#     def activate_without_storing(self, inputs):
#         # Version qui ne stocke pas dans self.outputs
#         positive = inputs >= 0
#         negative = ~positive
        
#         outputs = np.empty_like(inputs)
#         outputs[positive] = 1 / (1 + np.exp(-inputs[positive]))
#         exp_x = np.exp(inputs[negative])
#         outputs[negative] = exp_x / (1 + exp_x)
#         return outputs