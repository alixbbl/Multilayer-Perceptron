import pandas as pd
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from model.layer import DenseLayer
from model.optimizer import Optimizer
from model.loss import Loss_CategoricalCrossEntropy, Loss_BinaryCrossEntropy
from model.utils import print_network_structure

def target_encoder(y: pd.Series)->np.ndarray:
    """
        Converts labels in 0 (B) and 1 (M) for the training phase.
    """
    mapping = {'B': 0, 'M': 1}
    y_encoded = y.map(mapping).values
    return y_encoded

def data_loader(X: pd.DataFrame, y: np.ndarray, batch_size: int):
    """
        X: DataFrame, y: NumPy array (après target_encoder)
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]
        X_batch = X.iloc[batch_indices].values
        y_batch = y[batch_indices]
        yield X_batch, y_batch


# ****************************************** MULTILAYER PERCEPTRON *****************************************

class MLP:

    def __init__(self, n_inputs: int, n_neurons: List, n_output: int, loss, learning_rate):
        
        mlp_network = [n_inputs] + n_neurons + [n_output]
        print_network_structure(mlp_network)
        self.optimizer = Optimizer(learning_rate)
        self.n_inputs = n_inputs # nombre de features
        self.n_output = n_output # vaut 1 car sortie en classification binaire
        self.learning_rate = learning_rate
        
        if loss == "categoricalCrossentropy" :
            self.loss_function = Loss_CategoricalCrossEntropy()
        else:
            self.loss_function = Loss_BinaryCrossEntropy()

        self.layers = []
        for i in range(len(mlp_network) - 1):
            if i < len(mlp_network) - 2:
                activation = "reLu"
            else:
                activation = "SoftMax" if loss == "categoricalCrossentropy" else "Sigmoid"
            new_layer = DenseLayer(mlp_network[i], mlp_network[i + 1], activation)
            self.layers.append(new_layer) # rappel on doit avoir le nombre de neurones = output de la couche avant

    def feed_forward(self, X_batch: pd.DataFrame)->np.ndarray:
        # passer les inputs a travers le network
        # qppliquer les fonctions d'activation et calculer la sortie
        inputs = X_batch
        for layer in self.layers:
            layer.forward(inputs)
            layer.activation.activate(layer.outputs)
            inputs = layer.outputs
        return inputs # == predictions, on ne peut pas stocker donc on return

    # on calcule le gradient (derivee de la fonction de loss) pour le batch et on applique couche 
    # apres couche, la focntion backward qui va venir propager gradient et learning rate
    def backward_propagation(self, y_pred_batch, y_batch):
        grad_dOutputs = self.loss_function.compute_gradient(y_pred_batch, y_batch)
        for layer in reversed(self.layers):
            grad_dInputs = layer.backward(grad_dOutputs)
            grad_dOutputs = grad_dInputs
        
        for layer in reversed(self.layers):
            layer.weights = self.optimizer.step(layer.weights, layer.dWeights)
            layer.biases = self.optimizer.step(layer.biases, layer.dBiases)


    def train(self, X_train, y_train, epochs, batch_size):
        # Boucler sur toutes les epochs
        # A chaque epoch, faire un feed_forward + backpropagation
        y_train = target_encoder(y_train)
        loss_history = []
        
        for epoch in range(epochs):
            batches_loss = []
            for X_batch, y_batch in data_loader(X_train, y_train, batch_size):
                y_pred_batch = self.feed_forward(X_batch)
                # print(f'\nY_PRED_BATCH IS : \n{y_pred_batch}')
                loss_batch = self.loss_function.calculate_final_loss(y_pred_batch, y_batch) #=> a conserver en suivi 
                batches_loss.append(loss_batch)
                self.backward_propagation(y_pred_batch, y_batch)
            epoch_loss = np.mean( batches_loss)
            loss_history.append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        return loss_history

    def predict(self, X):
        pass
