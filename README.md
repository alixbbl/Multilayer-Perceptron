# MULTILAYER PERCEPTRON

## 🚀 LANCEMENT DES PROGRAMMES

Set up le virtual env et la config du projet :
```bash
source venv/bin/activate
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Commandes scripts :
```bash
python scripts/process_data.py --path_csv_to_read data/data.csv --select_features three
python scripts/train.py --layer 24 24 24 --epochs 20 --loss binaryCrossentropy --batch_size 32 --learning_rate 0.0314
python scripts/validate.py
```
---

## 🎯 OBJECTIF : DÉTECTION DE CELLULES CANCÉREUSES

Le dataset "Wisconsin Diagnostic Breast Cancer (WDBC)" a servi à des chercheurs à mettre au point un modèle pour identifier les cellules cancéreuses malignes.

### 📊 Baseline de référence
- **Méthode :** Multisurface Method-Tree (MSM-T) 
- **Performance :** 97.5% de précision
- **Features utilisées :** 3 seulement
  - Worst Area (pire aire)
  - Worst Smoothness (pire lissage) 
  - Mean Texture (texture moyenne)

**Notre défi :** Égaler ou dépasser ces performances avec un MLP !

---

## ✅ EXIGENCES 42

- [x] 2 couches cachées minimum
- [x] Architecture modulaire 
- [x] Activation softmax en sortie
- [x] Fonction de coût : log loss
- [x] Split train/validation
- [x] Métriques affichées à chaque epoch
- [x] Courbes d'apprentissage
- [x] Sauvegarde/chargement du modèle
- [x] Évaluation avec cross-entropy

### ⚠️ Incohérence dans l'énoncé
**Softmax** (quand on a 2 neurones) + **Binary Cross Entropy** (quand on a 1 neurone) 
→ **Solution :** Implémenter les deux architectures

---

## 🔄 PIPELINE D'IMPLÉMENTATION

### 1️⃣ PREPROCESSING DES DONNÉES

#### Dataset info
- **569 échantillons** (357 bénignes, 212 malignes)
- **30 features** disponibles
- **Aucune valeur manquante**
- **Classes déséquilibrées** (62.7% vs 37.3%)

#### Étapes de preprocessing
1. **Nettoyage** : Vérifier l'absence de NaN
2. **Split train/test** : Sans sklearn (implémentation manuelle)
3. **Sélection des features** :
   - Histogrammes pour features discriminantes
   - Matrice de corrélation pour éliminer redondance
   - Analyse de variance pour features informatives
4. **Standardisation** des données (recommandé pour un MLP)
5. **Encodage** de la target (0/1 ou one-hot)

### 2️⃣ ARCHITECTURE DU MLP

#### Structure générale
```
Input Layer (n_features) → Hidden Layer 1 → Hidden Layer 2 → Output Layer
```

#### Règles d'implémentation
- **Initialisation :** Poids aléatoires entre -1 et 1 (`np.random.randn() * 0.1`)
- **Biais :** Initialisés à zéro (`np.zeros((1, n_neurons))`)
- **Connectivité :** inputs(n) = neurons(n-1)

#### Fonctions d'activation
- **Hidden layers :** ReLU (`max(0, x)`)
- **Output layer :** Softmax ou Sigmoid selon l'architecture

### 3️⃣ ENTRAÎNEMENT

#### Forward Pass
```python
z = W·X + b  # Combinaison linéaire
a = activation(z)  # Activation
```

#### Backward Pass - Chaîne des gradients
```
dLoss/dW = dLoss/da × da/dz × dz/dW
```

**Pour chaque couche :**
1. Calculer `dLoss/da` (gradient de la loss)
2. Calculer `da/dz` (dérivée de l'activation)
3. Calculer `dz/dW` (gradient par rapport aux poids)
4. Mettre à jour : `W -= learning_rate * dLoss/dW`

#### Loss Functions
- **Binary Cross Entropy :** `-(y*log(p) + (1-y)*log(1-p))`
- **Categorical Cross Entropy :** `-Σ(y_true * log(y_pred))`

### 4️⃣ ÉVALUATION

#### Métriques principales

| Métrique | Formule | Usage |
|----------|---------|-------|
| **Accuracy** | (TP + TN) / Total | Performance globale |
| **Precision** | TP / (TP + FP) | "Quand je dis cancer, ai-je raison ?" |
| **Recall** | TP / (TP + FN) | "Est-ce que je trouve tous les cancers ?" |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Équilibre P/R |

#### Rappel : la matrice de confusion
```
              Prédiction
           |  B  |  M  |
Réalité B  | TN  | FP  |
        M  | FN  | TP  |
```

⚠️ **En médical : Minimiser les False Negatives** (cancers ratés)

#### LES SIGNES QUE LE MODELE NE MARCHE PAS CORRECTEMENT :
- **Overfitting :** train_accuracy >> val_accuracy
- **Underfitting :** des performances faibles partout
- **Instabilité :** des courbes qui oscillent

---

## 🎯 LES SCENARIOS POSSIBLES :

- **✅ Situation normale**
Train: 90% → Test: 87%   (Légère baisse, c'est normal)

- **⚠️ Overfitting**
Train: 99% → Test: 75%   (Grosse chute = mémorisation)

- **😱 Underfitting**
Train: 60% → Test: 58%   (Les deux faibles = modèle trop simple)

-**🎉 Situation idéale**
Train: 92% → Test: 91%   (Très proche = bon modèle)