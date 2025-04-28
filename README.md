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


# PREMIERE ETAPE : CREATION DU DATASET DE TEST

Exemple de fonctions pour creer son propre dataset de classification
points = le nombre de points de donnees a generer pour chaque classe
classes = la nombre de classes qu'on veut
On doit dns un premier temps nettoyer et separer le dataset fourni en deux parties, l'une pour entrainer le modele, la seconde pour le tester.
Usuellement on utilisera plutot train_test_split(), interdite ici.
 

# IMPLEMENTER LE MULTILAYER PERCEPTRON

La regle la plus importante ici est de s'assurer qu'on a toujours un nombre d'inputs de la couche n, egal au nombre de neurones de la couche n-1, sauf evidemment dans le cadre de l'input layer.
On initialise les weights au hasard mais en fixant des petites valeurs entre -1 et 1 pour s'assurer de la stabilite mathematiques du calcul.
n_inputs sera le nombre de features et n_neurons le nombre de neurones de la couche
=> Utiliser pn.randn pour produire des valeurs random (utiliser une seed).

Pour utiliser le biais, on va creer une colonne de 0 de la taille du nombre de neurones avec np.zeros ;
cette fonction attend normalement un seul argument, on met donc un tuple en argument afin de specifier les deux dimensions.
=> on lui donne donc le tuple des coordonnees => chaque neurone de la layer a ainsi son propre biais.

self.biases est donc de forme [[0.0.0....]] => 2D, on ne bosse que en 2D donc pas de [0.0...].


# LES FONCTIONS D'ACTIVATION

Elles permettent de rendre plus interpretables les resultats des fonctions de cout.

## La fonction Sigmoid

Elle permet de transformer la sortie lors de classification binaire, on l'utilise pour les sorties output layer a deux neurones. cf DSLR

## La fonction ReLU

Elle est plus simple et marche pour les hidden layers, donne 0 si faux et un float si vrai, cet aspect binaire est plus simple et
marche bien pour les couches cachees.

## La fonction SoftMax

La SoftMax est utilisee en sortie de reseau sur la output layer car elle va retourner des probabilites, ce qui est plus utile dans
une tache de classification multi-classes, d'autant qe l'obtention de probabilites permet de savoir a quel point le modele est
performant, ce qu'on ne pourrait pas calculer avec des resultats 0 et 1.
=> Dans la couche "output layer", elle transforme la sortie de vecteurs de floats de la derniere hidden layer en vecteurs de probas.

Elle peut donc servir a entrainer un MLP par backpropagation (ajuster les poids avec ReLU n'est pas faisable), car les probas sont
plus facilement interpretables et porteuses de nuances.
NB : elle utilise l'exponentielle + la normalisation pour assurer la stabilite mathematique du calcul.

# LA LOG LOSS : CATEGORICAL CROSS ENTROPY

L'entropie croisée est particulièrement bien adaptée aux problèmes de classification multi-classes.
Elle pénalise fortement les prédictions incorrectes, ce qui aide le modèle à apprendre à être plus précis.

On implémente une classe parente abstraite dont la méthode calculate va appeler le calcul de la loss sur chaque échantillon du dataset,

Ensuite on va en faire la moyenne et retourner le resultat. Cette architecture apporte de la modularite si on souhaite utiliser une autre fonction de loss que la Categorical Cross Entropy (Mean Squared Error ou Mean Value par exemple).

Dans la classe fille, on utilise la Categorical Cross Entropy,

Puisque le logarithme de 0 est infini, on doit supprimer les 0 des predictions, idem pour 1 (ces deux valeurs sont problématiques).
np.clip() : Cette fonction est utilisée pour limiter les valeurs d'un tableau NumPy à un intervalle donné. Elle prend trois arguments :

    Le tableau à clipper (y_pred).

    La valeur minimale (1e-7).

    La valeur maximale (1-1e-7).

np.clip() parcourt chaque élément du tableau y_pred et remplace toute valeur inférieure à 1e-7 par 1e-7. Elle remplace toute valeur supérieure à 1-1e-7 par 1-1e-7. Toutes les valeurs comprises entre ces deux bornes restent inchangées.

Le if /elif vise a verifier si on donne y_true sous forme de vecteur unidimensionnel [0, 1, 0, 0] si on a deux classes ou [2, 0, 0, 1]par exemple si on a 3 classes ; ou de matrice one-hot [[0, 1], [1, 0], [0, 1], [0, 1]].

### Principe d'indexation avancee :

La ligne de code :
`correct_confidences = y_pred_clipped[range(samples), y_true]`

Si y_true est [1, 2, 0] :

    Échantillon 0 : Classe 1, Probabilité = 0.6

    Échantillon 1 : Classe 2, Probabilité = 0.3

    Échantillon 2 : Classe 0, Probabilité = 0.2

Le range(samples) va renvoyer une sequence d'entiers de 0 a len(samples - 1).

On va pouvoir recuperer la proba de chaque prediction :
Si y_true contient [1, 2, 0], cela signifie que pour le premier échantillon, la classe correcte est la deuxième classe (indice 1), pour le deuxième échantillon c'est la troisième classe (indice 2), et pour le troisième échantillon c'est la première classe (indice 0).
 
y_pred_clipped pourrait être :

[
    [0.1, 0.6, 0.3],

    [0.3, 0.4, 0.3],

    [0.2, 0.5, 0.3]
]

La sélection avec l'indexation avancée récupère les valeurs [0.6, 0.3, 0.2], qui sont les probabilités que le modèle a assignées aux classes correctes selon y_true. On ne compare pas les probas des pred et des truths, c'est la log qui vient penaliser les pred faibles !

# OPTIMISATION DE LA LOSS => TRAINING DU MODELE MLP

Entrée brute -> Prédictions brutes -> Softmax -> Probabilités -> Calcul de la Loss -> Backpropagation
