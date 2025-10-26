# Détection de fraude bancaire (PySpark)

> Pipeline de classification en **PySpark / MLlib** sur le jeu **IEEE-CIS (Kaggle)** :
> ~590 k transactions, **≈ 400 variables** (anonymisées + identité partielle),
> **3.5 %** de fraudes, fort déséquilibre de classes.



---

##  Objectif
Construire un pipeline **distribué** et **reproductible** pour prédire `isFraud` à partir d’environ **400 variables hétérogènes**, avec :
- fusion et préparation des données (`train_transaction` + `train_identity`)  
- nettoyage ciblé et imputation des valeurs manquantes  
- *feature engineering* guidé par l’EDA 
- gestion du déséquilibre de classes (pondération)  
- benchmark : **Logistic Regression**, **Random Forest**, **GBT**  
- optimisation : tuning d’hyperparamètres & ajustement du seuil  
- visualisation et suivi exploratoire via **Apache Superset**


---

##  Stack
- **Spark / PySpark (MLlib)** pour le data prep & les modèles  
- **Python** : pandas, numpy, scipy (tests statistiques), matplotlib/seaborn
- **Superset** pour le tableau de bord EDA

---

##  Données
Compétition **IEEE-CIS Fraud Detection** (Kaggle).  
Les fichiers ne sont **pas** versionnés : placez‐les dans `data/` (ignoré par git).
- `train_transaction.csv`
- `train_identity.csv`

---

##  Pipeline (vue rapide)

1. **Fusion** `transaction ⟷ identity` (LEFT JOIN par `TransactionID`)  
2. **Nettoyage**
   - retrait des colonnes >90% manquants  
   - création d’indicateurs `has_X` (80–90% manquants)  
   - imputation : **médiane** (num), `"unknown"` (cat)
3. **Feature engineering**
   - dérivées : `log1p_TransactionAmt`, `C1_over_D1`, `day`, `is_weekend`…
   - flags métier : `is_product_C`, `is_credit_card`, `is_discover_card`, `is_high_amount`, `is_recent_activity`…
   - catégorielles : `StringIndexer` + `OneHotEncoder` (faible cardinalité) ; **frequency encoding** pour domaines email
   - assemblage : `VectorAssembler` (+ `StandardScaler` quand utile)
4. **Split** 80/20 (seed=42)  
5. **Déséquilibre** : **pondération de classes** via `weightCol`
6. **Modèles** : LR / RF / **GBT**  
7. **Sélection & tuning** : importances GBT, sous-échantillonnage pour grid restreinte (`maxDepth`, `maxIter`), **calibrage du seuil**  
8. **Évaluation** : ROC-AUC, PR-AUC, F1, Precision/Recall + courbes ROC & PR

---

##  Résultats (validation)

| Modèle | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.872 | — | 0.18 | 0.10 | 0.84 |
| Random Forest | 0.855 | — | 0.17 | 0.10 | 0.81 |
| **GBT (final)** | **0.954** | **0.74** | **0.69** | **0.72** | **0.67** |

**Choix final : GBT** (`maxDepth=10`, `maxIter=100`, **seuil=0.8**) — bon compromis *precision/recall* après ajout de features « anti-FP » et calibration du seuil.

> Détails et tableaux complets dans le rapport (`/rapports/…pdf`).  

---

##  Prise en main

### 0) Cloner

git clone https://github.com/elfahad98/ieee-fraud-pyspark.git
cd ieee-fraud-pyspark

