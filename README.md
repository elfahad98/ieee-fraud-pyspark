# Détection de fraude bancaire — Machine Learning avec (PySpark)

> Pipeline de classification **PySpark / MLlib** sur **IEEE-CIS (Kaggle)** :  
> ~590 k transactions, **≈ 400 variables disponibles** après fusion (`transaction` + `identity`),  
> **3,5 %** de fraudes (déséquilibre marqué), variables largement anonymisées.

<!-- Optionnel : ajoute la capture si tu la poses dans docs/ -->
<!-- ![Dashboard Superset (aperçu)](docs/superset_hero.png) -->

---

##  Objectif & périmètre

Construire un pipeline **distribué** et **reproductible** pour prédire `isFraud` sur un jeu de données
large et hétérogène, **sans analyser exhaustivement chaque variable** : l’EDA sert à **cibler un
sous-ensemble de variables réellement informatives**, puis on optimise le modèle (sélection,
tuning **manuel** et **calibration du seuil**) en tenant compte du **déséquilibre (~3,5 %)**.

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

> Placez les données dans `./data/` (répertoire ignoré par git).

---

## Résultats (validation)

| Modèle            | ROC-AUC | PR-AUC | F1    | Précision | Rappel |
|-------------------|:------:|:-----:|:-----:|:--------:|:------:|
| **GBT (optimisé)**| **0.954** | **0.74** | **0.692** | **0.78** | **0.64** |

**Modèle retenu** : `GBTClassifier` (**maxDepth=10**, **maxIter=100**, **seuil=0.845**).  
Réglages efficaces : **features ciblées**, **sélection**, **tuning**, **calibration du seuil**.


---

## Visualisations clés

| Figure | Commentaire rapide |
|:--|:--|
| ![PR curves](screenshots/optirapp.png) | **Précision–Rappel (val.)** : **PR-AUC ≈ 0,737** (*prévalence ≈ 3,5 %*) — bonne séparation malgré le fort déséquilibre. |
| ![Threshold](screenshots/Validation__GBT__threshold_curves.png) | **Courbes Precision/Recall/F1 vs seuil** : pic **F1=0.692** au **seuil=0.845** → compromis adopté. |
| ![Confusion](screenshots/Validation__GBT__confusion_matrix_t0.84.png) | **Matrice (val., t≈0.845)** : **TP=2 646**, **FP=743**, **FN=1 610**, **TN=113 109** → erreurs surtout côté rappel. |
| ![Gain](screenshots/Validation__GBT__cumulative_gain.png) | **Gain cumulatif** : en priorisant ~les meilleurs scores, une petite part de la population capte la majorité des fraudes. |
| ![Calibration](screenshots/Validation__GBT__calibration_curve.png) | **Calibration** : légère **sur-confiance** en milieu de gamme (courbe sous la diagonale) → à corriger si la proba est consommée telle quelle. |
| ![SHAP](screenshots/shap.png) | **Interprétabilité (SHAP, échantillon)** : `TransactionAmt`, `card1`, `log1p_D15`, `C13`, `V257` ressortent clairement. |


### Pistes d’amélioration
- **Calibration** (isotonic / Platt) sur un set dédié, puis rééval sur un **hold-out**.
- **Validation temporelle** (split par date) pour mieux gérer la dérive.
- **Features** robustes (fenêtres temporelles, ratios stables, encodages fréquence).
- **Suivi prod** : PR-AUC, PR@k, drift des features/labels, ré-entraînement périodique.


---


##  Prise en main

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
```

---

## 👤 Auteur

Projet réalisé par **COMBO El-Fahad** – Université de Caen (2025).  
Contact : `el-fahad.combo@etu.unicaen.fr`

---

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier `LICENSE`.
