from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update_weights(self, layer):
        pass

class SGD_optimizer(Optimizer):
    
    def update_weights(self, layer):
        layer.weights -= self.learning_rate * layer.dWeights
        layer.biases -= self.learning_rate * layer.dBiases

