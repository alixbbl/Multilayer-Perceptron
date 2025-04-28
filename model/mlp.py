import pandas as pd
import numpy as np
import argparse
from typing import List, Tuple
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import argparse

def create_data(points, classes):
    X = np.zeros((points*classes, 2)) # creation d'un tableau en 2D 
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 3)
plt.scatter(X[:,0], X[:,1])
plt.show()

# **************************************** LAYERS, ACTIVATION & LOSS **************************************

np.random.seed(0)

class Layer_dense:
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # on devrait faire l'inverse, on evite d'utiliser la .T => essayer avec! 
        self.biases = np.zeros((1, n_neurons)) # on fait toujours un tableau du nombre de neurones, un biais / neurone.
    
    def forward(self, inputs): # input peut etre de la data en entree de la input_layer ou les resultats de la cocuhe precedente
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_reLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

class Activation_Sigmoid:
    def forward(self, inputs):
        self.outputs = np.where(inputs >= 0, 1 / (1 + np.exp(-inputs)), np.exp(inputs) / (1 + np.exp(inputs)))

class Activation_SoftMax():
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        self.output = probabilities

# on a n echantillons ou samples, chaque echantillon va occasionner sa loss, donc la loss globale sera la moyenne de loss de tous les 
# samples, d'ou l'usage de np.mean(). Un sample sera un vecteur contenant le meme nbre de valeurs que de classes de sortie.
class Loss(ABC):
    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass
    def calculate(self, output, y):
        sample_losses = self.compute_loss(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss # permet de faire la moyenne de tous les echantillons du dataset quelle que soit la fonction d'activation

# la consigne demande que la Binary soit utilisee car on a que deux resultats possibles en sortie : M ou B
class Loss_BinaryCrossEntropy(Loss):
    def compute_loss(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        loss = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return loss

class Loss_CategoricalCrossEntropy(Loss):
    def compute_loss(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # on clip les valeurs de 0 et 1 dont les logarithmes vont faire crasher les calculs
        # si y_true est une liste d'indices de classes comme [1, 3, 3, 0, 2, 1] avec 0, 1, 2, 3 les ID des classes
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # si y_true est une matrice issue d'un one-hot encoding avec juste des 0 et 1 pour Faux/Vrai
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_likelihoods = -np.log(correct_confidences)
        return negative_likelihoods

# ****************************************** GRADIENT OPTIMIZATION *****************************************

# ajouter le suivi de la loss pour la partie graphique de la consigne
class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update_weights(self, weights, gradient):
        weights -= self.learning_rate * gradient
        return weights
    

# ****************************************** MULTILAYER PERCEPTRON *****************************************

class MLP:
    def __init__(self): # choix des n_neurons et n_inputs ici
        pass
    def backward_propagation(self):
        pass
    def feed_forward(self):
        pass

# **************************************************** MAIN *************************************************

X, y = create_data(100, 3) # on a ici 100 points et 3 classes, donc 300 echantillons

layer1 = Layer_dense(2, 3)
activation1 = Activation_Sigmoid()

layer2 = Layer_dense(3, 3)
activation2 = Activation_Sigmoid()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.outputs)
activation2.forward(layer2.output)

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.outputs, y)
print(loss)

# def main(parsed_args):
#     pass

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
    