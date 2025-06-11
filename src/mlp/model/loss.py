import numpy as np
from abc import ABC, abstractmethod

# ****************************************** LOSS FUNCTIONS ****************************************

class Loss(ABC):
    
    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass
    
    def calculate_final_loss(self, output: np.ndarray, y: np.ndarray):
        sample_losses = self.compute_loss(output, y)
        batch_loss = np.mean(sample_losses)
        return batch_loss

class Loss_BinaryCrossEntropy(Loss):
    def __init__(self, class_weight_0=1.0, class_weight_1=1.0):
        self.class_weight_0 = class_weight_0
        self.class_weight_1 = class_weight_1
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred = y_pred.flatten()
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        weights = y_true * self.class_weight_1 + (1 - y_true) * self.class_weight_0
        loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return weights * loss
    
    def compute_loss_gradient(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Calcule le gradient de la Binary Cross Entropy Loss avec pondération des classes
        """
        y_true = np.array(y_true, dtype=np.float64)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        
        m = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        grad_base = y_pred_clipped - y_true
        weights = y_true * self.class_weight_1 + (1 - y_true) * self.class_weight_0
        grad_weighted = grad_base * weights
        
        return grad_weighted

class Loss_CategoricalCrossEntropy(Loss):
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y_pred = y_pred.flatten()
        if len(y_true.shape) == 2:
            y_true_indices = np.argmax(y_true, axis=1)
        else:
            y_true_indices = y_true
        
        correct_confidences = y_pred_clipped[range(len(y_pred)), y_true_indices]
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def compute_loss_gradient(self, y_pred: np.ndarray, y_true: np.ndarray):
        m = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            y_true_onehot = np.zeros_like(y_pred_clipped)
            y_true_onehot[np.arange(m), y_true.astype(int)] = 1
            y_true = y_true_onehot
        grad = (y_pred_clipped - y_true) / m
        return grad