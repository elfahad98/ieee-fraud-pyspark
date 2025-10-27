# Détection de fraude bancaire (PySpark)

> Pipeline de classification **PySpark / MLlib** sur **IEEE-CIS (Kaggle)** :  
> ~590k transactions, **≈ 400+ variables disponibles** après fusion (`transaction` + `identity`),  
> **3,5 %** de fraudes (déséquilibre marqué), variables largement anonymisées.

<!-- Optionnel : ajoute la capture si tu la poses dans docs/ -->
<!-- ![Dashboard Superset (aperçu)](docs/superset_hero.png) -->

---

##  Objectif & périmètre

Construire un pipeline **distribué** et **reproductible** pour prédire `isFraud` sur un jeu de données
large et hétérogène, **sans analyser exhaustivement chaque variable** : l’EDA sert à **cibler un
sous-ensemble de variables réellement informatives**, puis on optimise le modèle (sélection,
tuning **manuel** et **calibration du seuil**) en tenant compte du **déséquilibre (≈ 3,5 %)**.

---

##  Méthodologie (6 étapes)

1) **Prétraitement initial**  
   - Chargement `train_transaction` et `train_identity`  
   - **Fusion** (LEFT JOIN) sur `TransactionID`  
   - **EDA “brute”** : structure, typage, taux de fraude, détection d’anomalies simples

2) **Nettoyage ciblé**  
   - Suppression des colonnes peu informatives (beaucoup de NaN / constantes)  
   - Création d’indicateurs **`has_X`** pour colonnes très incomplètes  
   - **Imputation** : médiane (numériques), `"unknown"` (catégorielles)

3) **Analyse exploratoire avancée (EDA)**  
   - Relations avec `isFraud` (ex. `TransactionAmt`, `D1`, `C1`, `ProductCD`, `card4`, `card6`, domaines email…)  
   - Recherche de **patterns**, effets de seuil, corrélations, **groupes à risque**  
   - L’EDA **guide** la suite : on **cible** des familles/variables utiles (on ne balaie **pas** “les 400” une par une)

4) **Feature Engineering (FE)**  
   - Dérivées & ratios : `log1p_TransactionAmt`, `C1/D1`, `day`, `is_weekend`…  
   - **Flags métier** : `is_product_C`, `card4_card6`, `is_high_amount`, `is_recent_activity`, `is_recent_intense`, etc.  
   - **Encodage** :  
     - faible cardinalité → `StringIndexer` + `OneHotEncoder`  
     - cardinalité élevée (emails) → **frequency encoding**  
   - **Assemblage** : `VectorAssembler` (+ `StandardScaler` si pertinent)

5) **Modélisation & déséquilibre**  
   - **Split** 80/20 (seed fixe)  
   - **Pondération** des classes via `weightCol`  
   - **Benchmark** : Logistic Regression (baseline), Random Forest, **GBT**

6) **Optimisation & sélection**  
   - **Sélection** de variables (importances GBT : globales / par familles / `Vxx` + filtre simple)  
   - **Tuning manuel** ciblé (ex. `maxDepth`, `maxIter`) sur sous-échantillon  
   - **Calibration du seuil** pour optimiser le compromis précision/rappel

---

##  Stack

- **Spark / PySpark (MLlib)** : préparation, pipeline, modèles  
- **Python** : pandas, numpy, **scipy** (tests statistiques : χ², etc.), matplotlib/seaborn  
- **Superset** : visualisation EDA & suivi des indicateurs (**pas** en temps réel)  
- *(Optionnel)* **SHAP** sur un échantillon Pandas (proxy sklearn) pour interprétabilité

---

##  Données

Compétition **IEEE-CIS Fraud Detection** (Kaggle).  
Les fichiers **ne sont pas versionnés** (licence) :
- `train_transaction.csv`
- `train_identity.csv`

---

##  Résultats (validation)

| Modèle               | ROC-AUC | PR-AUC | F1   | Précision | Rappel |
|----------------------|:------:|:-----:|:----:|:--------:|:------:|
| **GBT (optimisé)**      | **0.954** | **0.74** | **0.69** | **0.72** | **0.67** |

**Modèle retenu** : `GBTClassifier` (**maxDepth = 10**, **maxIter = 100**, **seuil = 0.8**).  
Le gain vient des **features ciblant les faux positifs**, de la **sélection de variables**, du **tuning**
et de la **calibration du seuil**.

---

## Prise en main

```bash
# 0) Cloner
git clone https://github.com/elfahad98/ieee-fraud-pyspark.git
cd ieee-fraud-pyspark

# 1) Environnement (ex. venv)
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt

# 2) Données (non versionnées) : déposer dans ./data/
# data/train_transaction.csv
# data/train_identity.csv

# 3) Lancer
# - Notebook principal :
#   notebooks/fraud_detection_modeling1.ipynb
