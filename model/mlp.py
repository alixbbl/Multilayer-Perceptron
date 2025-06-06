import pandas as pd
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from model.layer import DenseLayer
from model.optimizer import Optimizer
from model.loss import Loss_CategoricalCrossEntropy, Loss_BinaryCrossEntropy
from model.utils import print_network_structure, save_config
import constants

def target_encoder(y, categorical=False):
    """
        Converts labels: 
        - Binary: 'B'->0, 'M'->1
        - Categorical: one-hot encoding
    """
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(y[0], str):
        mapping = {'B': 0, 'M': 1}
        y_encoded = pd.Series(y).map(mapping).values
    else:
        y_encoded = y
    if categorical:
        one_hot = np.zeros((len(y_encoded), 2))
        one_hot[np.arange(len(y_encoded)), y_encoded.astype(int)] = 1
        return one_hot
    return y_encoded.astype(int)

def data_loader(X: pd.DataFrame, y: np.ndarray, batch_size: int):
    """
        Data loader with shuffling.
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
        constants.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        save_config(mlp_network, loss, learning_rate)
        
        self.optimizer = Optimizer(learning_rate)
        self.n_inputs = n_inputs # nombre de features
        self.n_output = n_output # vaut 1 car sortie en classification binaire
        self.learning_rate = learning_rate
        self.loss_type = loss
        
        if loss == "categoricalCrossentropy" :
            self.loss_function = Loss_CategoricalCrossEntropy()
        else:
            self.loss_function = Loss_BinaryCrossEntropy()

        self.layers = []
        for i in range(len(mlp_network) - 1):
            if i < len(mlp_network) - 2:
                activation = "relu"
            else:
                activation = "SoftMax" if loss == "categoricalCrossentropy" else "Sigmoid"
            
            new_layer = DenseLayer(mlp_network[i], mlp_network[i + 1], activation)
            self.layers.append(new_layer) # rappel on doit avoir le nombre de neurones = output de la couche avant

    def feed_forward(self, X_batch: np.ndarray) -> np.ndarray:
        """
        
        """
        inputs = X_batch
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward_propagation(self, y_pred_batch: np.ndarray, y_batch: np.ndarray) -> None:
        """
            Feed the model backward to adjust the weights and biases og the different layers,
            based on the gradients ratio : dL/dW into .... COMPLETER.
        """
        # on recupere le gradient de la loss par rapports aux predictions dL/doutputs
        grad_dOutputs = self.loss_function.compute_loss_gradient(y_pred_batch, y_batch)
        # backpropagation ici
        for layer in reversed(self.layers):
            grad_dOutputs = layer.backward(grad_dOutputs)
        # maj des poids
        for layer in self.layers:
            layer.weights -= self.optimizer.learning_rate * layer.dWeights
            layer.biases -= self.optimizer.learning_rate * layer.dBiases

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Returns a Numpy array of predictions based on training weights and biases.
        """
        y_pred = self.feed_forward(X)
        if self.loss_type == "categoricalCrossentropy":
            return np.argmax(y_pred, axis=1)
        else:
            return (y_pred >= 0.5).astype(int).flatten()

    def save_parameters(self, filepath):
        """"
            Save the MLP weights and biases into a NPZ file.
        """
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'layer_{i}_weights'] = layer.weights
            params[f'layer_{i}_biases'] = layer.biases
        np.savez(filepath, **params)

    def train(self, X_train, y_train, epochs, batch_size):
        categorical = (self.loss_type == "categoricalCrossentropy")
        y_train_encoded = target_encoder(y_train, categorical=categorical)
        loss_history = []
        accuracy_history = []

        for epoch in range(epochs):
            batches_loss = []
            
            for X_batch, y_batch in data_loader(X_train, y_train_encoded, batch_size):
                y_pred_batch = self.feed_forward(X_batch)
                loss_batch = self.loss_function.calculate_final_loss(y_pred_batch, y_batch) # indicative
                batches_loss.append(loss_batch)
                self.backward_propagation(y_pred_batch, y_batch)
            
            epoch_loss = np.mean(batches_loss)
            loss_history.append(epoch_loss)
            
            y_pred_train = self.predict(X_train.values if hasattr(X_train, 'values') else X_train)
            y_true_binary = target_encoder(y_train, categorical=False)  # Toujours binaire pour accuracy
            epoch_accuracy = np.mean(y_pred_train == y_true_binary)
            accuracy_history.append(epoch_accuracy)

            y_pred_proba = self.feed_forward(X_train.values[:100])  # Les 100 premiers
            print(f"Prédictions: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

        return loss_history, accuracy_history