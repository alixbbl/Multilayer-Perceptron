# MULTILAYER PERCEPTRON : BÂTIR LE MODÈLE

### Architecture des couches

La règle la plus importante est de s'assurer qu'on a toujours un **nombre d'inputs de la couche n, égal au nombre de neurones de la couche n-1**, sauf évidemment dans le cadre de l'input layer.

Exemple :
```python
Input Layer:    features (30) → Hidden Layer 1 (16 neurones)
Hidden Layer 1: 16 inputs    → Hidden Layer 2 (8 neurones) 
Hidden Layer 2: 8 inputs     → Output Layer (2 neurones pour softmax ou 1 pour sigmoid)
```

### Initialisation des poids et biais

L'initialisation doit garantir la stabilite mathematique du modele, on utilise deux methodes ici : 
- **Xavier** : si la fonction d'activation de la couche est reLu.
```python
# Initialisation des poids
weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
```

- **HE** : si la fonction est Sigmoid ou Softmax. 
```python
# Initialisation des poids
weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
```

Pour le biais, on va créer une **matrice de 0** de la taille du nombre de neurones avec `np.zeros`. Cette fonction attend normalement un seul argument, on met donc **un tuple en argument** afin de spécifier les deux dimensions.
```python
# Initialisation des biais  
self.biases = np.zeros((1, n_neurons))
```
⚠️ **Important :** `self.biases` est donc de forme `[[0.0, 0.0, 0.0...]]` => **2D**, on ne bosse qu'en 2D donc pas de `[0.0...]`.

---
## 1. FONCTIONS DE LOSS

### Binary Cross Entropy (BCE)
**Usage** : Classification binaire (2 classes et 1 seul neurone en sortie)

**Formule** :
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```
- `y` = vraie classe (0 ou 1)
- `ŷ` = probabilité prédite (0 ≤ ŷ ≤ 1)

**Gradient** :
```
∂L/∂ŷ = (ŷ - y) / [ŷ(1-ŷ)]
```
**Simplification avec Sigmoid** : `∂L/∂ŷ = ŷ - y`

### Categorical Cross Entropy (CCE)
**Usage** : Classification multi-classes (n > 2 classes, avec n neurones en sortie)

**Formule** :
```
L = -∑ᵢ yᵢ·log(ŷᵢ)
```
- `yᵢ` = vraie classe (one-hot encoding)
- `ŷᵢ` = probabilité prédite pour la classe i

**Gradient** :
```
∂L/∂ŷᵢ = -yᵢ/ŷᵢ
```
**Simplification avec Softmax** : `∂L/∂ŷ = ŷ - y`

---

## 2. FONCTIONS D'ACTIVATION

### Sigmoid
**Formule** : `σ(z) = 1 / (1 + e^(-z))`

**Dérivée** : `σ'(z) = σ(z) × (1 - σ(z))`

**Usage** : Couche de sortie pour classification binaire

### ReLU (Rectified Linear Unit)
**Formule** : `ReLU(z) = max(0, z)`

**Dérivée** : 
```
ReLU'(z) = 1 si z > 0
         = 0 si z ≤ 0
```

**Usage** : Couches cachées (résout le problème du gradient qui disparaît)

### Softmax
**Formule** : `softmax(zᵢ) = e^(zᵢ) / ∑ⱼ e^(zⱼ)`

**Propriétés** :
- Sortie entre 0 et 1
- Somme des probabilités = 1
- Amplification des différences

**Usage** : Couche de sortie pour classification multi-classes

---

## 3. ARCHITECTURE D'UN MLP

### Structure générale
```
Input → Hidden Layer(s) → Output Layer → Loss
  x    →     z₁, a₁     →    z₂, a₂    →  L
```

**Notation** :
- `zₗ` = sortie brute de la couche l : `zₗ = Wₗ·aₗ₋₁ + bₗ`
- `aₗ` = sortie activée : `aₗ = activation(zₗ)`

### Types de couches
- **Input** : Reçoit les données (pas d'activation)
- **Hidden** : Apprentissage des représentations (ReLU)
- **Output** : Prédictions finales (Sigmoid/Softmax)

---

## 4. BACKPROPAGATION - LA CHAÎNE DE DÉRIVATION

### Principe fondamental : Chain Rule
Pour calculer `∂L/∂W`, on décompose :
```
∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W
```

### Étape 1 : Calcul du gradient de la loss
**Point de départ** : `∂L/∂a` (gradient par rapport aux activations de sortie)
- BCE + Sigmoid : `∂L/∂a = ŷ - y`
- CCE + Softmax : `∂L/∂a = ŷ - y`

### Étape 2 : Gradient de l'activation
**Calcul** : `∂a/∂z` (dérivée de la fonction d'activation)
- Sigmoid : `∂a/∂z = a(1-a)`
- ReLU : `∂a/∂z = 1 si z > 0, sinon 0`

**Résultat** : `∂L/∂z = ∂L/∂a × ∂a/∂z`

### Étape 3 : Gradients des paramètres
**Pour les poids** : `∂z/∂W = aₗ₋₁` (activations de la couche précédente)
```
∂L/∂W = ∂L/∂z × ∂z/∂W = ∂L/∂z × aₗ₋₁
```

**Pour les biais** : `∂z/∂b = 1`
```
∂L/∂b = ∂L/∂z × 1 = ∂L/∂z
```

### Étape 4 : Propagation vers la couche précédente
**Gradient pour la couche précédente** : `∂z/∂aₗ₋₁ = W`
```
∂L/∂aₗ₋₁ = ∂L/∂z × W
```

### Flow complet de la backpropagation
```
1. ∂L/∂a_output ← Gradient de la loss
2. ∂L/∂z_output ← ∂L/∂a × ∂a/∂z (dérivée activation)
3. ∂L/∂W_output ← ∂L/∂z × inputs de cette couche
4. ∂L/∂b_output ← ∂L/∂z
5. ∂L/∂a_hidden ← ∂L/∂z × W (propager vers couche précédente)
6. Répéter 2-5 pour chaque couche (de la sortie vers l'entrée)
```

---

## 5. OPTIMISATION

### Gradient Descent (GD)
**Principe** : Mise à jour dans la direction opposée au gradient
```
W ← W - η × ∂L/∂W
b ← b - η × ∂L/∂b
```
- `η` = learning rate (taux d'apprentissage)

### Techniques avancées

**Gradient Clipping** : Limiter la norme des gradients
```
Si ||∇W|| > threshold :
    ∇W ← ∇W × (threshold / ||∇W||)
```

**Early Stopping** : Arrêter quand la validation ne s'améliore plus
- Surveiller la loss de validation
- Arrêter si pas d'amélioration pendant N epochs

---

## 6. CONCEPTS CLÉS

### Entraînement par batches
- **Epoch** : Un passage complet sur tout le dataset
- **Batch** : Sous-ensemble du dataset traité simultanément
- **Mini-batch gradient descent** : Compromis entre vitesse et stabilité

### Overfitting vs Underfitting
- **Overfitting** : Le modèle mémorise les données d'entraînement
- **Underfitting** : Le modèle est trop simple pour capturer les patterns
- **Solution** : Validation set + early stopping

### Normalisation des données
**Standardisation** : `x_std = (x - μ) / σ`
- Accélère la convergence
- Évite que certaines features dominent