# Détection de fraude bancaire (PySpark)

> Pipeline de classification **PySpark / MLlib** sur le jeu **IEEE-CIS (Kaggle)** :  
> **590 540** transactions, **≈ 434 colonnes après fusion** (`transaction` + `identity`), **3,5 %** de fraudes, variables largement **anonymisées** et identité partielle. :contentReference[oaicite:0]{index=0}

---

##  Objectif & périmètre
Construire un pipeline distribué et **reproductible** pour prédire `isFraud` à partir d’un **grand nombre de variables disponibles (~400)**, en **ciblant uniquement** un **sous-ensemble informatif** identifié par l’EDA et par la **sélection de variables** (importances GBT par familles/id/`Vxx` + filtre `Vxx` > 0,01). :contentReference[oaicite:1]{index=1}  
Concrètement :
- **Fusion** `train_transaction` + `train_identity` (LEFT JOIN sur `TransactionID`). :contentReference[oaicite:2]{index=2}  
- **Nettoyage ciblé** : retrait colonnes > 90 % NaN, création d’indicateurs `has_X` (80–90 % manquants), **imputation** (médiane num, `"unknown"` cat). :contentReference[oaicite:3]{index=3}  
- **Feature engineering guidé par l’EDA** : dérivées (`log1p_TransactionAmt`, `C1/D1`, `day`, `is_weekend`), **flags métier** (`is_product_C`, `is_credit_card`, `is_discover_card`, `card4_card6`, `is_recent_activity`, `is_recent_intense`, etc.). :contentReference[oaicite:4]{index=4}  
- **Encodage** : `StringIndexer` + `OneHotEncoder` (faible cardinalité), **frequency encoding** pour email-domains, assemblage `VectorAssembler`, scaling si utile. :contentReference[oaicite:5]{index=5}  
- **Déséquilibre** (3,5 % fraude) géré par **pondération des classes** (`weightCol`). :contentReference[oaicite:6]{index=6}  
- **Modèles** benchmark : Logistic Regression, Random Forest, **GBT**. :contentReference[oaicite:7]{index=7}  
- **Optimisation** : sélection de variables, **tuning** (`maxDepth`, `maxIter`) sur sous-échantillon 40 %, **calibrage du seuil** de décision. :contentReference[oaicite:8]{index=8}

---

## ⚙️ Stack
- **Spark / PySpark (MLlib)** : préparation, pipeline, modèles  
- **Python** : pandas, numpy, **scipy** (tests stats : Mann-Whitney, χ²), matplotlib/seaborn  
- **Superset** : EDA & visualisation (captures dans `docs/`)  
- *(Optionnel)* **scikit-learn** : modèle proxy pour **SHAP** (interprétabilité). :contentReference[oaicite:9]{index=9}

---

## 🧱 Pipeline (très synthétique)
1) Fusion & typage → 2) Nettoyage (NaN, `has_X`, imputation) →  
3) EDA avancée (cible vs `TransactionAmt`, `D1`, `C1`, `ProductCD`, `card4`, `card6`, etc.) → 4) FE (ratios/flags/temps) →  
5) Encodage & assemblage → 6) Split 80/20 (seed=42), **pondération** → 7) Benchmark → 8) Sélection + tuning + **seuil**. :contentReference[oaicite:10]{index=10}

---

## 🔢 Résultats finaux (validation)
**Modèle retenu** : `GBTClassifier` (**maxDepth=10**, **maxIter=100**, **seuil=0.8**).  
**Validation** : **ROC-AUC = 0.9543**, **F1 = 0.6948**, **Precision = 0.7194**, **Recall = 0.6718**. :contentReference[oaicite:11]{index=11}

> Le gain vient de : **features ciblant les faux positifs**, sélection de variables, **tuning** + **calibration du seuil** (précision ~14 % → ~72 %). :contentReference[oaicite:12]{index=12}

---

## 🚀 Reproduire
```bash
# 0) Cloner
git clone https://github.com/elfahad98/ieee-fraud-pyspark.git
cd ieee-fraud-pyspark

# 1) Environnement (ex. venv)
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) Données (non versionnées)
# Déposer depuis Kaggle :
# data/train_transaction.csv
# data/train_identity.csv

# 3) Lancer
# - Notebook principal : notebooks/fraud_detection_modeling1.ipynb
# - ou script (si fourni plus tard) : python scripts/train_gbt.py --data_dir data
