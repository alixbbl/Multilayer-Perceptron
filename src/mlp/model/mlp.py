import pandas as pd
import numpy as np
import json
from typing import List
from .layer import DenseLayer
from .optimizer import SGD_optimizer
from .loss import Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy
from .utils import print_network_structure, save_config
from .early_stopper import Early_Stopper
from .metrics import Metrics
from mlp.config import MODEL_OUTPUT

def target_encoder(y, categorical=False)->np.ndarray:
    """
        Convert labels: Binary 'B'->0, 'M'->1 or categorical one-hot.
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    if isinstance(y.iloc[0], str):
        mapping = {'B': 0, 'M': 1}
        y_encoded = y.map(mapping).values
    else:
        y_encoded = y.values

    if categorical:
        one_hot = np.zeros((len(y_encoded), 2))
        one_hot[np.arange(len(y_encoded)), y_encoded.astype(int)] = 1
        return one_hot

    return y_encoded

# ****************************************** MULTILAYER PERCEPTRON *****************************************

class MLP:
    
    def __init__(self, n_inputs: int, n_neurons: List, 
                 n_output: int, loss, learning_rate, 
                 save_name):
        
        mlp_network = [n_inputs] + n_neurons + [n_output]
        print_network_structure(mlp_network)
        save_config(mlp_network, loss, learning_rate)
        
        self.optimizer = SGD_optimizer(learning_rate)
        self.n_inputs = n_inputs
        self.n_output = n_output
        self.config = mlp_network
        self.learning_rate = learning_rate
        self.loss_type = loss
        self.save_name = save_name
        
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
            self.layers.append(new_layer)

    def _feed_forward(self, X_batch: np.ndarray) -> np.ndarray:
        """
            Calculate the outputs of every layer based of the outputs of the previous ones.
        """
        inputs = X_batch
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def _update_weights(self):
        """
            Update weights using chosen optimizer.
        """
        for layer in self.layers:
            self.optimizer.update_weights(layer)
    
    def _backward_propagation(self, y_pred_batch: np.ndarray, y_batch: np.ndarray) -> None:
        """
            Compute gradients via backpropagation.
            Includes gradient clipping (max norm=5.0) to prevent exploding gradients.
        """
        grad_dOutputs = self.loss_function.compute_loss_gradient(y_pred_batch, y_batch)
        
        for layer in reversed(self.layers):
            grad_dOutputs = layer.backward(grad_dOutputs)
        max_gradient_norm = 5.0
        
        for layer in self.layers:
            grad_norm = np.linalg.norm(layer.dWeights)
            if grad_norm > max_gradient_norm:
                layer.dWeights = layer.dWeights * (max_gradient_norm / grad_norm)

            grad_norm = np.linalg.norm(layer.dBiases)
            if grad_norm > max_gradient_norm:
                layer.dBiases = layer.dBiases * (max_gradient_norm / grad_norm)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
            Returns a Numpy array of predictions based on training weights and biases.
            These predictions are always biaised, so training accuracy cannot be an 
            reliable performance indicator.
        """
        y_pred = self._feed_forward(X)
        if self.loss_type == "categoricalCrossentropy":
            return np.argmax(y_pred, axis=1)
        else:
            return (y_pred >= 0.5).astype(int).flatten()

    def save_weights(self, filepath: str):
        """"
            Save the MLP weights and biases into a NPZ file.
        """
        weigths = {}
        for i, layer in enumerate(self.layers):
            weigths[f'layer_{i}_weights'] = layer.weights
            weigths[f'layer_{i}_biases'] = layer.biases
        np.savez(filepath, **weigths)

    def save_training_params_into_JSON(self, 
                                       save_name, 
                                       loss_history, 
                                       accuracy_history, 
                                       recall_history, 
                                       precision_history):
        """"
            Save the all MLP training scores into a JSON.
        """
        results = {
            "model_name" : save_name,
            "config" : self.config,
            "learning_rate" : self.learning_rate,
            "loss_history" : loss_history,
            "accuracy_history" : accuracy_history,
            "recall_history" : recall_history,
            "precision_history" : precision_history,
        }
        with open(f'outputs/training_results_{save_name}.json', 'w') as f:
            json.dump(results, f)


    def train(self, X_train: pd.DataFrame, y_train: pd.Series, epochs: int, 
              batch_size: int):
        """
            Train the model accordingly to the parameters.
        """
        categorical = (self.loss_type == "categoricalCrossentropy")
        y_train_encoded = target_encoder(y_train, categorical=categorical)
        y_true_binary = target_encoder(y_train, categorical=False)
        X_train_array = X_train.values
        
        loss_history = []
        accuracy_history = []
        recall_history = []
        precision_history = []
        
        early_stopper = Early_Stopper()
        
        for epoch in range(epochs):
            total_loss = 0
            total_samples = 0
            
            for start in range(0, len(X_train_array), batch_size):
                end = min(start + batch_size, len(X_train_array))
                X_batch = X_train_array[start:end]
                y_batch = y_train_encoded[start:end]
                
                y_pred_batch = self._feed_forward(X_batch)
                loss_batch = self.loss_function.calculate_final_loss(y_pred_batch, y_batch)
                
                batch_size_actual = len(X_batch)
                total_loss += loss_batch * batch_size_actual
                total_samples += batch_size_actual
                
                self._backward_propagation(y_pred_batch, y_batch)
                self._update_weights()
            
            epoch_loss = total_loss / total_samples
            loss_history.append(epoch_loss)
            
            y_pred_train = self.predict(X_train_array)
            metrics = Metrics(y_true_binary, y_pred_train)
            
            epoch_accuracy = metrics._calculate_accuracy()
            epoch_precision = metrics._calculate_precision()
            epoch_recall = metrics._calculate_recall()
            
            accuracy_history.append(epoch_accuracy)
            precision_history.append(epoch_precision)
            recall_history.append(epoch_recall)
            
            should_stop = early_stopper(epoch_loss)
            
            if epoch % 3 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - "
                    f"Accuracy: {epoch_accuracy:.4f} - Precision: {epoch_precision:.4f} - "
                    f"Recall: {epoch_recall:.4f}")
            
            if should_stop:
                print(f"🛑 Early stopping at epoch {epoch+1} (patience={early_stopper.patience})")
                break
        
        if hasattr(self, 'save_name') and self.save_name:
            self.save_training_params_into_JSON(
                self.save_name, 
                loss_history, 
                accuracy_history, 
                precision_history, 
                recall_history
            )
        
        return loss_history, accuracy_history,\
                precision_history, recall_history