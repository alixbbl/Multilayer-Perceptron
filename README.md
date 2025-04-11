# MULTILAYER PERCEPTRON

👉 POUR REACTIVER LE VIRTUAL ENV :
    'python3 -m venv env'
    'source env/bin/activate'
    'pip install -r requirements.txt'

👉 POUR LANCER LA PREPARATION DES DONNEES :
    'python -m data_processing.process_data'
Depuis la racine du projet.

## Definition et concepts-clefs

Un multilayer perceptron est un reseau de neurones artificiel possedant au moins trois couches :
- couche d'entree, 
- couche intermediaire ou cachee,
- couche de sortie.
Chaque couche possede un nombre variable de neurones, qui se transmettent les informations de la couche 
d'entree vers la couche de sortie, soit en feedforward. Ce modele a ete invente en 1957 par Franck 
Rosenblatt - le perceptron etait lors de son invention, monocouche.
Il est performant pour modeliser les reations entre les donnees lineaires et non-lineaires.

# Feedforward (propagation avant)

💡 L’info entre dans le réseau, traverse les couches, et on obtient une prédiction.

À chaque couche :

    On fait un produit matriciel + un biais → Z = W·X + b

    On applique une fonction d’activation (ex : sigmoid, relu, softmax)

# Calcul de la perte (loss) ou fonction de cout

Une fois qu’on a la prédiction, on la compare à la valeur réelle (la vérité terrain) pour calculer l’erreur, avec une fonction de coût :

    En classification binaire : log loss (ou binary cross-entropy)
    En multi-classes : categorical cross-entropy

# Backpropagation (propagation arrière)

Objectif : ajuster les poids du réseau pour qu’il s’améliore.

Il faut donc :

1- Calculer l’erreur finale
2- Puis propager cette erreur en sens inverse dans le réseau pour :
    - calculer les gradients (pentes de la fonction de coût par rapport aux poids)
    - mettre à jour les poids avec la descente de gradient

👉 Cette étape utilise la dérivée des fonctions d’activation (ex: dérivée de sigmoid, softmax…).


## ATTENTES DU SUJET : 

Éléments obligatoires :

2 couches cachées min.	                ✅
Modularité (fichier ou args)	        ✅
Activation sigmoid et softmax	        ✅
Fonction de coût : log loss (binaire)	✅
Split du dataset (train/val)	        ✅
Affichage des métriques à chaque epoch	✅
Courbes d’apprentissage (train & val)	✅
Sauvegarde du modèle (topo + poids)	    ✅
Chargement du modèle en prédiction	    ✅
Évaluation avec cross-entropy	        ✅

# Concevoir le programme

L'input layer est composee de X neurones, X etant le nombre de features du dataset.