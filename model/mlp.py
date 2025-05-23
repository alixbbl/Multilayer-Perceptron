import pandas as pd
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from model.layer import DenseLayer
from model.activation import Activation_SoftMax, Activation_Sigmoid, Activation_reLU
from model.loss import Loss_CategoricalCrossEntropy, Loss_BinaryCrossEntropy


def data_loader(X, y, batch_size):

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]


# ****************************************** MULTILAYER PERCEPTRON *****************************************

class MLP:

    def __init__(self, n_inputs: int, n_neurons: List, n_output: int, loss, learning_rate):
        
        mlp_network = [n_inputs] + n_neurons + [n_output]
        print(f'You are building a network with these layers : {mlp_network}')
        
        self.n_inputs = n_inputs # nombre de features
        self.n_output = n_output # vaut 1 car sortie en classification binaire
        self.learning_rate = learning_rate
        
        if loss == "loss_CategoricalCrossEntropy" :
            self.loss_function = Loss_CategoricalCrossEntropy()
        else:
            self.loss_function = Loss_BinaryCrossEntropy()

        self.layers = []
        for i in range(len(mlp_network) - 1):
            self.layers.append(DenseLayer(mlp_network[i], mlp_network[i + 1])) # rappel on doit avoir le nombre de neurones = output de la couche avant
        for layer in self.layers[:-1]:
            layer.activation = Activation_reLU()
        if loss == "loss_CategoricalCrossEntropy":
            self.layers[-1].activation = Activation_SoftMax()
        else:
            self.layers[-1].activation = Activation_Sigmoid() # avec binary

    def feed_forward(self, X_batch: pd.DataFrame)->pd.DataFrame:
        # passer les inputs a travers le network
        # qppliquer les fonctions d'activation et calculer la sortie
        inputs = X_batch 
        for layer in self.layers:
            layer.forward(inputs)
            layer.activation(layer.outputs)
            inputs = layer.outputs
        return inputs # == predictions, on ne peut pas stocker donc on return

    # on calcule le gradient (derivee de la fonction de loss) pour le batch et on applique couche 
    # apres couche, la focntion backward qui va venir propager gradient et learning rate
    def backward_propagation(self, y_pred_batch, y_batch):
        # calculer la loss, puis les gradients avec la fonction de cout 
        # Maj des poids et biais => se fait dans backward de DenseLayer
        grad_dOutputs = self.loss_function.compute_gradient(y_pred_batch, y_batch)
        for layer in reversed(self.layers):
            grad_dInputs = layer.backward(grad_dOutputs)
            grad_dOutputs = grad_dInputs
        for layer in reversed(self.layers):
            layer.update_weights(self.learning_rate)

    def train(self, X_train, y_train, epochs, batch_size):
        # Boucler sur toutes les epochs
        # A chaque epoch, faire un feed_forward + backpropagation
        for epoch in range(epochs):
            for X_batch, y_batch in data_loader(X_train, y_train, batch_size):
                y_pred_batch = self.feed_forward(X_batch)
                loss_batch = self.loss_function.calculate_final_loss(y_pred_batch, y_batch)
                # cf loss_batch a conserver pour suivi de la visualisation de l'apprentissage => on 
                # se sert de loss_batch comme indicateur et pas dans la retropropagation 
                self.backward_propagation(y_pred_batch, y_batch, self.learning_rate)

    def predict(self, X):
        pass
