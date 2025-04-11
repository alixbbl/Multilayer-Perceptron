import pandas as pd
import numpy as np
import argparse
from typing import List, Tuple

class MLP():
    
    def __init__(self):
        pass
    
    def ft_relu(self, x):
        """
            "Rectified Linear Unit" is an activation function used in MLP hidden layers.
            Replaces all negative values with 0 and keeps the positive ones unchanged.
            x is a numpy array or input list, returns numpy array with ReLU applied to each element.
        """
        return np.maximum(0, x)
    
    def ft_softmax(self, z):
        """
            This function is an activation function, used in classification tasks with more than 
            two classes (in opposite to sigmoid which is used for binary classification).
            Applies on a predictions vector (one-D numpy array) and returns a probabilities vector.
        """
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def log_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
        """
            Computes the binary cross-entropy (log loss) for logistic regression.
            Args:
                y_true (np.array): Array of true labels (0 or 1).
                y_pred (np.array): Array of predicted probabilities (between 0 and 1).
            Returns the average log loss (float).
        """
        epsilon = 1e-15  # To avoid log(0), which is undefined
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Keeps values in the range (0,1)
        loss_value = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss_value