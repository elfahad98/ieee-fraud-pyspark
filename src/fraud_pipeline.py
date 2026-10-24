import json
from pathlib import Path

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StandardScalerModel,
    StringIndexer,
    VectorAssembler,)
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql.functions import (
    approx_count_distinct,avg,broadcast,col,concat_ws,count,exp,
    expr,floor,isnan, length,lit,log1p,
    mean,percentile_approx,stddev,when,sum as spark_sum,
)


# =========================
# constants
# =========================

IDENTITY_COLUMNS = ["DeviceType", "id_01", "id_30", "id_31", "DeviceInfo"]

TEST_COLS_TO_DROP = [
    "dist2", "D7", "id_07", "id_08", "id_18", "id_21", "id_22",
    "id_23", "id_24", "id_25", "id_26", "id_27", "TransactionID",
    "DeviceInfo",
]

TEST_HAS_COLUMNS = [
    "D6", "D8", "D9", "D12", "D13", "D14", "V138", "V139", "V140", "V141",
    "V142", "V143", "V144", "V145", "V146", "V147", "V148", "V149", "V150",
    "V151", "V152", "V153", "V154", "V155", "V156", "V157", "V158", "V159",
    "V160", "V161", "V162", "V163", "V164", "V165", "V166", "V322", "V323",
    "V324", "V325", "V326", "V327", "V328", "V329", "V330", "V331", "V332",
    "V333", "V334", "V335", "V336", "V337", "V338", "V339", "id_03", "id_04",
    "id_09", "id_10", "id_14", "id_32", "id_33", "id_34",
]

LOG_FEATURE_COLUMNS = ["TransactionAmt", "C1_sur_D1", "D1"]

VARS_TO_ENCODE = [
    "ProductCD", "card4", "card6", "DeviceType", "M4", "id_15", "id_16",
    "id_28", "id_29", "id_35", "id_36", "id_37", "id_38", "M1", "M2", "M3",
    "M5", "M6", "id_12", "id_34",
]

VARS_TO_FREQUENCY_ENCODE = [
    "P_emaildomain", "R_emaildomain", "card4_card6", "M789_combo",
    "id_30", "id_31",
    "uid_simple", "uid_full",
]

VARS_TO_TARGET_ENCODE = ["P_emaildomain", "R_emaildomain"]

SUSPICIOUS_VALUES = [4.77068, 5.42495, 5.81711, 5.14749, 3.93182, 6.31896]
SUSPICIOUS_MARGIN = 0.05
SUSPICIOUS_ADDR1 = [204.0, 325.0]
SUSPICIOUS_RDOMAINS = ["unknown", "anonymous.com"]
SUSPICIOUS_PDOMAINS = ["unknown", "yahoo.com", "anonymous.com"]
GROUP_STATS_DEFAULT_GROUPS = ["card1"]
GROUP_STATS_VALUE_COLUMN = "TransactionAmt"
FN_RECOVERY_GROUPS = [
    ("card1", "risk_card1"),
    ("risk_key_product_card6", "risk_product_card6"),
    ("risk_key_product_device", "risk_product_device"),
    ("risk_key_device_r", "risk_device_r"),
]


# =========================
# filesystem helpers
# =========================

def ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))



def read_parquet_map(spark, path):
    """
    Lit une petite map Parquet Spark en ciblant directement les fichiers part-*.
    Sur Windows, Hadoop peut parfois refuser le listing du dossier local.
    """
    path = Path(path)
    part_files = sorted(path.glob("part-*.parquet"))
    if part_files:
        return spark.read.parquet(*[part.as_posix() for part in part_files])
    return spark.read.parquet(path.as_posix())


# =========================
# analysis helpers
# =========================

ADDED_BINARY_FEATURE_CANDIDATES = [
    "is_recent_activity",
    "is_recent_intense",
    "is_high_amount",
    "is_email_mismatch",
    "is_missing_D15",
    "is_product_C",
    "is_credit_card",
    "is_discover_card",
    "is_fn_productW_unknown_device",
    "is_fn_productW_unknown_recipient",
    "is_fn_debit_unknown_device",
    "is_fn_debit_unknown_recipient",
    "is_fn_m4_m0_productW",
    "is_fn_unknown_device_unknown_recipient",
    "is_amount_suspicious",
    "is_fp_prone_product",
    "is_productW_suspicious_amount",
    "is_fp_prone_device",
    "is_suspicious_recipient_email",
    "is_productW_unknown_device",
    "is_productW_unknown_recipient",
    "is_debit_unknown_device",
]


def added_feature_family(feature_name):
    """
    Regroupe les features ajoutées par famille pour faciliter l'interprétation.
    Cette fonction est utilisée uniquement en analyse, elle ne modifie pas les features du modèle.
    """
    if feature_name.startswith("risk_") and not feature_name.startswith("risk_key_"):
        return "risk_stats_train_only"
    if feature_name.startswith("is_fn_"):
        return "fn_recovery_flags"
    if feature_name.startswith("is_fp_") or feature_name in {
        "is_productW_suspicious_amount",
        "is_productW_unknown_device",
        "is_productW_unknown_recipient",
        "is_debit_unknown_device",
        "is_suspicious_recipient_email",
        "is_amount_suspicious",
    }:
        return "error_pattern_flags"
    if feature_name.endswith("_freq"):
        return "frequency_encoding"
    if feature_name in {
        "is_recent_activity",
        "is_recent_intense",
        "is_high_amount",
        "is_email_mismatch",
        "is_missing_D15",
        "is_product_C",
        "is_credit_card",
        "is_discover_card",
    }:
        return "manual_baseline_flags"
    return "autres"


def profile_binary_added_features(df, dataset_name, columns, label_col="isFraud"):
    """
    Calcule la fréquence, le taux de fraude et le lift des flags binaires ajoutés.
    """
    import pandas as pd

    total = df.count()
    global_rate = float(df.agg(avg(label_col).alias("fraud_rate")).first()["fraud_rate"] or 0.0)
    if not columns:
        return pd.DataFrame()

    agg_exprs = []
    for feature in columns:
        feature_as_double = col(feature).cast("double")
        agg_exprs.append(spark_sum(feature_as_double).alias(f"{feature}__n"))
        agg_exprs.append(
            spark_sum(feature_as_double * col(label_col).cast("double")).alias(f"{feature}__frauds")
        )

    stats = df.agg(*agg_exprs).first().asDict()
    rows = []
    for feature in columns:
        n = float(stats.get(f"{feature}__n") or 0.0)
        frauds = float(stats.get(f"{feature}__frauds") or 0.0)
        fraud_rate = frauds / n if n else 0.0
        rows.append(
            {
                "dataset": dataset_name,
                "feature": feature,
                "family": added_feature_family(feature),
                "n_flag_1": int(n),
                "share_pct": round(100 * n / total, 3) if total else 0.0,
                "fraud_rate_pct": round(100 * fraud_rate, 3),
                "global_fraud_rate_pct": round(100 * global_rate, 3),
                "lift_vs_global": round(fraud_rate / global_rate, 3) if global_rate else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_added_feature_analysis(
    train_df,
    valid_df,
    feat_importance,
    feature_selection_dir,
    plots_dir,
    metadata_dir,
    top_n=20,
):
    """
    Produit les tableaux et graphiques d'interprétation des features ajoutées.

    Important : cette fonction ne réentraîne aucun modèle et ne modifie pas les DataFrames.
    Elle exploite seulement `feat_importance` déjà calculé dans le notebook 07.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    feature_selection_dir = Path(feature_selection_dir)
    plots_dir = Path(plots_dir)
    metadata_dir = Path(metadata_dir)
    ensure_dirs(feature_selection_dir, plots_dir, metadata_dir)

    added_importance = feat_importance.copy()
    added_importance["added_family"] = added_importance["feature"].apply(added_feature_family)
    added_importance = added_importance[added_importance["added_family"] != "autres"].copy()
    added_importance = added_importance.sort_values("importance", ascending=False)

    added_family_importance = (
        added_importance.groupby("added_family", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    added_importance.to_csv((feature_selection_dir / "added_feature_importance.csv").as_posix(), index=False)
    added_family_importance.to_csv(
        (feature_selection_dir / "added_feature_family_importance.csv").as_posix(),
        index=False,
    )

    plt.figure(figsize=(10, 7))
    top_added = added_importance.head(top_n).iloc[::-1]
    plt.barh(top_added["feature"], top_added["importance"], color="#3FA77A", edgecolor="#246B4D")
    plt.xlabel("Importance")
    plt.title("Top variables ajoutées utiles au modèle")
    plt.grid(axis="x", linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig((plots_dir / "added_feature_importance_top20.png").as_posix(), bbox_inches="tight")
    plt.show()

    binary_added_cols = [c for c in ADDED_BINARY_FEATURE_CANDIDATES if c in train_df.columns]
    binary_feature_profile = pd.concat(
        [
            profile_binary_added_features(train_df, "train", binary_added_cols),
            profile_binary_added_features(valid_df, "valid", binary_added_cols),
        ],
        ignore_index=True,
    )

    binary_feature_profile = binary_feature_profile.sort_values(
        ["dataset", "lift_vs_global", "n_flag_1"],
        ascending=[True, False, False],
    )
    binary_feature_profile.to_csv(
        (feature_selection_dir / "added_binary_feature_fraud_rates.csv").as_posix(),
        index=False,
    )

    valid_binary_profile = binary_feature_profile[binary_feature_profile["dataset"] == "valid"].head(15).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(
        valid_binary_profile["feature"],
        valid_binary_profile["lift_vs_global"],
        color="#F0A33A",
        edgecolor="#9A641F",
    )
    plt.axvline(1.0, color="#444444", linestyle="--", linewidth=1)
    plt.xlabel("Lift du taux de fraude vs moyenne valid")
    plt.title("Variables binaires ajoutées : sur-risque observé sur valid")
    plt.grid(axis="x", linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig((plots_dir / "added_binary_feature_lift_valid.png").as_posix(), bbox_inches="tight")
    plt.show()

    summary_payload = {
        "top_added_features": added_importance.head(15).to_dict(orient="records"),
        "added_family_importance": added_family_importance.to_dict(orient="records"),
        "binary_feature_profile_valid_top": (
            binary_feature_profile[binary_feature_profile["dataset"] == "valid"]
            .head(15)
            .to_dict(orient="records")
        ),
    }
    save_json(metadata_dir / "added_feature_analysis_summary.json", summary_payload)

    return {
        "added_importance": added_importance,
        "added_family_importance": added_family_importance,
        "binary_feature_profile": binary_feature_profile,
    }


# =========================
# data loading / merge
# =========================

def load_train_merged(spark, data_root):
    """
    Charge les datasets train transaction / identity puis les fusionne.
    """
    data_root = Path(data_root)
    df_transaction = spark.read.csv(
        (data_root / "train_transaction.csv").as_posix(),
        header=True,
        inferSchema=True,
    )
    df_identity = spark.read.csv(
        (data_root / "train_identity.csv").as_posix(),
        header=True,
        inferSchema=True,
    )
    return df_transaction.join(df_identity, on="TransactionID", how="left")


def load_test_merged(spark, data_root):
    """
    Charge les datasets test transaction / identity puis les fusionne.
    """
    data_root = Path(data_root)
    test_trans = spark.read.csv(
        (data_root / "test_transaction.csv").as_posix(),
        header=True,
        inferSchema=True,
    )
    test_id = spark.read.csv(
        (data_root / "test_identity.csv").as_posix(),
        header=True,
        inferSchema=True,
    )
    return test_trans.join(test_id, on="TransactionID", how="left")


# =========================
# data cleaning
# =========================

def clean_column_names(df):
    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.replace("-", "_"))
    return df


def add_has_identity(df):
    return df.withColumn(
        "has_identity",
        when(
            col("DeviceType").isNotNull()
            | col("id_01").isNotNull()
            | col("id_30").isNotNull()
            | col("id_31").isNotNull()
            | col("DeviceInfo").isNotNull(),
            1,
        ).otherwise(0),
    )


def clean_train_dataframe(df, drop_threshold=0.9, has_lower=0.8, has_upper=0.9):
    """
    Nettoie le train et retourne aussi les artefacts utiles au rejouage sur le test.
    """
    total_rows = df.count()
    null_df = df.select([(count(when(col(c).isNull(), c)) / total_rows).alias(c) for c in df.columns])
    null_ratios = null_df.first().asDict()

    cols_to_drop = [name for name, rate in null_ratios.items() if rate > drop_threshold]
    for required in ["TransactionID", "DeviceInfo"]:
        if required not in cols_to_drop and required in df.columns:
            cols_to_drop.append(required)

    df = df.drop(*cols_to_drop)

    null_df = df.select([(count(when(col(c).isNull(), c)) / total_rows).alias(c) for c in df.columns])
    null_ratios = null_df.first().asDict()
    has_columns = [
        name
        for name, rate in null_ratios.items()
        if has_lower <= rate <= has_upper and name not in IDENTITY_COLUMNS
    ]
    for col_name in has_columns:
        df = df.withColumn(f"has_{col_name}", when(col(col_name).isNotNull(), 1).otherwise(0))

    numeric_cols = [c for c, t in df.dtypes if t in ["double", "int"] and c != "isFraud"]
    null_df_num = df.select([(count(when(col(c).isNull(), c)) / total_rows).alias(c) for c in numeric_cols])
    null_ratios_num = null_df_num.first().asDict()
    numeric_null_columns = [k for k, v in null_ratios_num.items() if v > 0]
    median_exprs = [percentile_approx(col(c), 0.5).alias(c) for c in numeric_null_columns]
    numeric_fill_values = df.select(median_exprs).first().asDict() if median_exprs else {}
    if numeric_fill_values:
        df = df.fillna(numeric_fill_values)

    string_cols = [c for c, t in df.dtypes if t == "string"]
    null_df_str = df.select([(count(when(col(c).isNull(), c)) / total_rows).alias(c) for c in string_cols])
    null_ratios_str = null_df_str.first().asDict()
    string_null_columns = [k for k, v in null_ratios_str.items() if v > 0]
    string_fill_values = {colname: "unknown" for colname in string_null_columns}
    if string_fill_values:
        df = df.fillna(string_fill_values)

    artifacts = {
        "drop_threshold": drop_threshold,
        "has_range": [has_lower, has_upper],
        "cols_to_drop": cols_to_drop,
        "has_columns": has_columns,
        "numeric_fill_values": numeric_fill_values,
        "string_fill_columns": string_null_columns,
    }
    return df, artifacts


def add_missing_inference_columns(df, numeric_columns=None, string_columns=None):
    """
    Ajoute les colonnes attendues par la pipeline d'inference quand un CSV
    entrant ne contient pas toute la partie identity ou certains champs rares.
    """
    numeric_columns = set(numeric_columns or [])
    string_columns = set(string_columns or [])

    for col_name in sorted(string_columns):
        if col_name not in df.columns:
            df = df.withColumn(col_name, lit("unknown"))

    for col_name in sorted(numeric_columns - string_columns):
        if col_name not in df.columns:
            df = df.withColumn(col_name, lit(0.0))

    return df


def clean_test_df(
    test_df,
    cols_to_drop=None,
    cols_has_to_create=None,
    numeric_fill_values=None,
    string_fill_columns=None,
):
    cols_to_drop = cols_to_drop or TEST_COLS_TO_DROP
    cols_has_to_create = cols_has_to_create or TEST_HAS_COLUMNS
    total_rows = test_df.count()

    test_df = test_df.drop(*[c for c in cols_to_drop if c in test_df.columns])

    for col_name in cols_has_to_create:
        if col_name in test_df.columns:
            test_df = test_df.withColumn(f"has_{col_name}", when(col(col_name).isNotNull(), 1).otherwise(0))

    if numeric_fill_values is None:
        numeric_cols = [c for c, t in test_df.dtypes if t in ["double", "int"]]
        null_df_num = test_df.select([(count(when(col(c).isNull(), c)) / total_rows).alias(c) for c in numeric_cols])
        null_ratios_num = null_df_num.first().asDict()
        numeric_null_columns = [k for k, v in null_ratios_num.items() if v > 0]
        median_exprs = [percentile_approx(col(c), 0.5).alias(c) for c in numeric_null_columns]
        numeric_fill_values = test_df.select(median_exprs).first().asDict() if median_exprs else {}

    numeric_fill_values = {
        colname: value
        for colname, value in (numeric_fill_values or {}).items()
        if colname in test_df.columns and value is not None
    }
    if numeric_fill_values:
        test_df = test_df.fillna(numeric_fill_values)

    if string_fill_columns is None:
        string_cols = [c for c, t in test_df.dtypes if t == "string"]
        null_df_str = test_df.select([(count(when(col(c).isNull(), c)) / total_rows).alias(c) for c in string_cols])
        null_ratios_str = null_df_str.first().asDict()
        string_fill_columns = [k for k, v in null_ratios_str.items() if v > 0]

    string_fill_values = {
        colname: "unknown"
        for colname in (string_fill_columns or [])
        if colname in test_df.columns
    }
    if string_fill_values:
        test_df = test_df.fillna(string_fill_values)

    return test_df


# =========================
# feature engineering
# =========================

def add_baseline_features(df):
    df = df.withColumn("is_recent_activity", (col("D1") == 0).cast("int"))
    df = df.withColumn("is_high_amount", (col("TransactionAmt") > 1000).cast("int"))
    df = df.withColumn("is_product_C", (col("ProductCD") == "C").cast("int"))
    df = df.withColumn("is_credit_card", (col("card6") == "credit").cast("int"))
    df = df.withColumn("is_discover_card", (col("card4") == "discover").cast("int"))
    df = df.withColumn("card4_card6", concat_ws("_", col("card4"), col("card6")))
    df = df.withColumn("day", (col("TransactionDT") / 86400).cast("int"))
    df = df.withColumn("is_weekend", expr("int((day % 7) IN (5, 6))"))
    df = df.withColumn(
        "C1_sur_D1",
        when(col("D1") == 0, col("C1")).otherwise(col("C1") / (col("D1") + 1e-4)),
    )
    return df


def add_log_features(df, cols_log=None):
    cols_log = cols_log or LOG_FEATURE_COLUMNS
    for col_name in cols_log:
        df = df.withColumn(f"log1p_{col_name}", log1p(col(col_name)))
    return df


def add_post_model_features(df):
    df = df.withColumn("is_email_mismatch", (col("P_emaildomain") != col("R_emaildomain")).cast("int"))
    df = df.withColumn("card_id", concat_ws("_", col("card1"), col("card2"), col("card3"), col("card5")))
    df = df.withColumn("card_id_length", length(col("card_id")))
    df = df.withColumn("log1p_D15", log1p(col("D15")))
    df = df.withColumn("Amt_sur_D15", col("log1p_TransactionAmt") / (col("D15") + 1e-3))
    df = df.withColumn("log1p_Amt_sur_D15", log1p(col("Amt_sur_D15")))
    df = df.withColumn("log1p_D15", when(col("log1p_D15").isNull(), 0.0).otherwise(col("log1p_D15")))
    df = df.withColumn(
        "log1p_Amt_sur_D15",
        when(col("log1p_Amt_sur_D15").isNull(), 0.0).otherwise(col("log1p_Amt_sur_D15")),
    )
    df = df.withColumn("is_missing_D15", (col("D15").isNull() | isnan(col("D15"))).cast("int"))
    return df


def add_recent_intense_feature(df, threshold=2.5):
    df = df.withColumn(
        "is_recent_intense",
        ((col("log1p_C1_sur_D1") > threshold) & (col("is_recent_activity") == 1)).cast("int"),
    )
    df = df.withColumn(
        "log1p_recent_intense_score",
        when(col("is_recent_activity") == 1, col("log1p_C1_sur_D1")).otherwise(lit(0.0)),
    )
    return df


def add_temporal_features(df):
    # Variables temporelles pour capter la fraude nocturne et le rythme des transactions.
    if "TransactionDT" not in df.columns:
        return df
    df = df.withColumn(
        "TransactionHour",
        floor((col("TransactionDT") % lit(86400)) / lit(3600)).cast("int"),
    )
    df = df.withColumn(
        "is_night_transaction",
        ((col("TransactionHour") < 6) | (col("TransactionHour") >= 22)).cast("int"),
    )
    df = df.withColumn("log1p_TransactionHour", log1p(col("TransactionHour").cast("double")))
    return df


def add_uid_full(df):
    # Construction de l'identifiant client (UID) : card1 + addr1 + ecart_jours(D1, DateTransaction).
    for required in ("card1", "addr1", "D1", "TransactionDT"):
        if required not in df.columns:
            return df
    day = floor(col("TransactionDT") / lit(86400)).cast("int")
    card_tok = when(col("card1").isNull(), lit("missing")).otherwise(col("card1").cast("string"))
    addr_tok = when(col("addr1").isNull(), lit("missing")).otherwise(col("addr1").cast("string"))
    d1gap_tok = when(col("D1").isNull(), lit("na")).otherwise((col("D1") - day).cast("int").cast("string"))
    return df.withColumn("uid_full", concat_ws("_", card_tok, addr_tok, d1gap_tok))


def add_m789_combo(df):
    return df.withColumn("M789_combo", concat_ws("_", col("M7"), col("M8"), col("M9")))


def add_uid_simple(df):
    if "card1" not in df.columns or "addr1" not in df.columns:
        return df

    card1_token = when(col("card1").isNull(), lit("missing")).otherwise(col("card1").cast("string"))
    addr1_token = when(col("addr1").isNull(), lit("missing")).otherwise(col("addr1").cast("string"))
    return df.withColumn("uid_simple", concat_ws("_", card1_token, addr1_token))


def add_fn_recovery_flags(df):
    """
    Flags ciblés sur les profils observés dans les false negatives.

    Ces variables restent volontairement simples pour éviter d'ajouter trop de
    bruit : elles indiquent surtout les combinaisons ProductCD=W, device/email
    inconnus, debit et M4=M0 qui ressortaient dans l'analyse d'erreurs.
    """
    df = df.withColumn(
        "is_fn_productW_unknown_device",
        when((col("ProductCD") == "W") & (col("DeviceType") == "unknown"), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_fn_productW_unknown_recipient",
        when((col("ProductCD") == "W") & (col("R_emaildomain") == "unknown"), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_fn_debit_unknown_device",
        when((col("card6") == "debit") & (col("DeviceType") == "unknown"), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_fn_debit_unknown_recipient",
        when((col("card6") == "debit") & (col("R_emaildomain") == "unknown"), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_fn_m4_m0_productW",
        when((col("M4") == "M0") & (col("ProductCD") == "W"), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_fn_unknown_device_unknown_recipient",
        when((col("DeviceType") == "unknown") & (col("R_emaildomain") == "unknown"), 1).otherwise(0),
    )
    return df


def add_fn_recovery_keys(df):
    product = when(col("ProductCD").isNull(), lit("unknown")).otherwise(col("ProductCD").cast("string"))
    card6 = when(col("card6").isNull(), lit("unknown")).otherwise(col("card6").cast("string"))
    device = when(col("DeviceType").isNull(), lit("unknown")).otherwise(col("DeviceType").cast("string"))
    r_email = when(col("R_emaildomain").isNull(), lit("unknown")).otherwise(col("R_emaildomain").cast("string"))

    df = df.withColumn("risk_key_product_card6", concat_ws("_", product, card6))
    df = df.withColumn("risk_key_product_device", concat_ws("_", product, device))
    df = df.withColumn("risk_key_device_r", concat_ws("_", device, r_email))
    return df


def _add_card1_amount_behavior(df, value_col=GROUP_STATS_VALUE_COLUMN):
    if value_col not in df.columns or "risk_card1_amt_mean" not in df.columns:
        return df

    eps = lit(1e-3)
    df = df.withColumn(
        "risk_amt_over_card1_mean",
        log1p(col(value_col) / (col("risk_card1_amt_mean") + eps)),
    )

    if "risk_card1_amt_std" in df.columns:
        df = df.withColumn(
            "risk_amt_z_card1",
            (col(value_col) - col("risk_card1_amt_mean")) / (col("risk_card1_amt_std") + eps),
        )
    return df


def fit_and_apply_fn_recovery_train_valid(
    train_df,
    valid_df,
    stats_root,
    target_col="isFraud",
    value_col=GROUP_STATS_VALUE_COLUMN,
    smoothing=100.0,
    group_specs=None,
):
    """
    Apprend de petites statistiques de risque sur train uniquement puis les
    applique à valid.

    Les `fraud_rate` sont lissés avec le taux global du train pour éviter que
    les catégories rares créent du sur-apprentissage.
    """
    stats_root = Path(stats_root)
    ensure_dirs(stats_root)
    group_specs = group_specs or FN_RECOVERY_GROUPS

    train_df = add_fn_recovery_keys(add_fn_recovery_flags(train_df))
    valid_df = add_fn_recovery_keys(add_fn_recovery_flags(valid_df))

    stats = (
        train_df.select(target_col)
        .agg(
            count("*").alias("total"),
            spark_sum(when(col(target_col) == 1, 1).otherwise(0)).alias("frauds"),
        )
        .first()
    )
    total = int(stats["total"])
    frauds = int(stats["frauds"] or 0)
    global_rate = frauds / total if total else 0.0
    created_feature_cols = [
        "is_fn_productW_unknown_device",
        "is_fn_productW_unknown_recipient",
        "is_fn_debit_unknown_device",
        "is_fn_debit_unknown_recipient",
        "is_fn_m4_m0_productW",
        "is_fn_unknown_device_unknown_recipient",
    ]
    available_groups = []

    for group_col, prefix in group_specs:
        if group_col not in train_df.columns:
            continue

        available_groups.append(group_col)
        rate_col = f"{prefix}_fraud_rate"
        count_col = f"{prefix}_count_log"
        agg_exprs = [
            count("*").alias("__cnt"),
            spark_sum(when(col(target_col) == 1, 1).otherwise(0)).alias("__frauds"),
        ]

        if group_col == "card1" and value_col in train_df.columns:
            agg_exprs.extend(
                [
                    avg(value_col).alias("risk_card1_amt_mean"),
                    stddev(value_col).alias("risk_card1_amt_std"),
                ]
            )

        risk_df = train_df.groupBy(group_col).agg(*agg_exprs)
        risk_df = risk_df.withColumn(
            rate_col,
            (col("__frauds") + lit(global_rate * smoothing)) / (col("__cnt") + lit(smoothing)),
        )
        risk_df = risk_df.withColumn(count_col, log1p(col("__cnt")))
        risk_df = risk_df.drop("__cnt", "__frauds")
        risk_df = risk_df.fillna({name: 0.0 for name in risk_df.columns if name != group_col})
        risk_df.write.mode("overwrite").parquet((stats_root / group_col).as_posix())

        train_df = train_df.join(broadcast(risk_df), on=group_col, how="left")
        valid_df = valid_df.join(broadcast(risk_df), on=group_col, how="left")

        fill_values = {rate_col: global_rate, count_col: 0.0}
        if group_col == "card1":
            fill_values.update({"risk_card1_amt_mean": 0.0, "risk_card1_amt_std": 0.0})

        train_df = train_df.fillna(fill_values)
        valid_df = valid_df.fillna(fill_values)

        created_feature_cols.extend([rate_col, count_col])
        if group_col == "card1":
            created_feature_cols.extend(["risk_card1_amt_mean", "risk_card1_amt_std"])

    train_df = _add_card1_amount_behavior(train_df, value_col=value_col)
    valid_df = _add_card1_amount_behavior(valid_df, value_col=value_col)
    for feature_name in ["risk_amt_over_card1_mean", "risk_amt_z_card1"]:
        if feature_name in train_df.columns:
            created_feature_cols.append(feature_name)

    created_feature_cols = list(dict.fromkeys(created_feature_cols))
    return train_df, valid_df, available_groups, created_feature_cols




def apply_fn_recovery_maps(
    df,
    spark,
    stats_root,
    group_specs=None,
    value_col=GROUP_STATS_VALUE_COLUMN,
):
    """
    Applique au test les maps FN recovery apprises sur train dans le notebook 05.
    """
    stats_root = Path(stats_root)
    group_specs = group_specs or FN_RECOVERY_GROUPS

    df = add_fn_recovery_keys(add_fn_recovery_flags(df))

    for group_col, prefix in group_specs:
        if group_col not in df.columns:
            continue

        stats_path = stats_root / group_col
        if not stats_path.exists():
            continue

        risk_df = read_parquet_map(spark, stats_path)
        output_cols = [name for name in risk_df.columns if name != group_col]
        duplicate_outputs = [name for name in output_cols if name in df.columns]
        if duplicate_outputs:
            df = df.drop(*duplicate_outputs)
        fill_map = {name: 0.0 for name in output_cols}
        df = df.join(broadcast(risk_df), on=group_col, how="left")
        df = df.fillna(fill_map)

    return _add_card1_amount_behavior(df, value_col=value_col)


def _group_stat_feature_names(group_col, value_col=GROUP_STATS_VALUE_COLUMN):
    return [
        f"{group_col}_count",
        f"{value_col}_{group_col}_mean",
        f"{value_col}_{group_col}_std",
    ]


def add_group_behavior_features(df, value_col=GROUP_STATS_VALUE_COLUMN):
    """
    Construit quelques features dérivées des statistiques de groupe
    sans multiplier fortement la dimension.
    """
    eps = lit(1e-3)

    if value_col in df.columns and f"{value_col}_card1_mean" in df.columns:
        ratio_card1 = col(value_col) / (col(f"{value_col}_card1_mean") + eps)
        df = df.withColumn("log1p_amt_over_card1_mean", log1p(ratio_card1))

    return df


def add_false_positive_features(df):
    condition = None
    for value in SUSPICIOUS_VALUES:
        lower = value - SUSPICIOUS_MARGIN
        upper = value + SUSPICIOUS_MARGIN
        current = (col("log1p_TransactionAmt") >= lower) & (col("log1p_TransactionAmt") <= upper)
        condition = current if condition is None else condition | current

    df = df.withColumn("is_amount_suspicious", when(condition, 1).otherwise(0))
    df = df.withColumn("is_fp_prone_product", when(col("ProductCD") == "W", 1).otherwise(0))
    df = df.withColumn(
        "is_productW_suspicious_amount",
        when((col("ProductCD") == "W") & (col("log1p_TransactionAmt").between(3.9, 4.9)), 1).otherwise(0),
    )
    df = df.withColumn("is_fp_prone_device", when(col("DeviceType") == "unknown", 1).otherwise(0))
    df = df.withColumn(
        "is_suspicious_recipient_email",
        when(col("R_emaildomain").isin(SUSPICIOUS_RDOMAINS), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_productW_unknown_device",
        when((col("ProductCD") == "W") & (col("DeviceType") == "unknown"), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_productW_unknown_recipient",
        when((col("ProductCD") == "W") & (col("R_emaildomain") == "unknown"), 1).otherwise(0),
    )
    df = df.withColumn(
        "is_debit_unknown_device",
        when((col("card6") == "debit") & (col("DeviceType") == "unknown"), 1).otherwise(0),
    )
    return df


def fit_and_apply_group_aggregations_train_valid(
    train_df,
    valid_df,
    stats_root,
    group_cols=None,
    value_col=GROUP_STATS_VALUE_COLUMN,
):
    """
    Calcule quelques statistiques de groupe légères sur train_df uniquement,
    puis les applique à train_df et valid_df.

    On reste volontairement minimal :
    - groupes : `card1` et `uid_simple = card1 + addr1`
    - variable agrégée : `TransactionAmt`
    """
    stats_root = Path(stats_root)
    ensure_dirs(stats_root)

    group_cols = group_cols or GROUP_STATS_DEFAULT_GROUPS
    if "uid_simple" in group_cols:
        train_df = add_uid_simple(train_df)
        valid_df = add_uid_simple(valid_df)
    available_group_cols = [group_col for group_col in group_cols if group_col in train_df.columns]
    created_feature_cols = []

    if value_col not in train_df.columns:
        return train_df, valid_df, available_group_cols, created_feature_cols

    for group_col in available_group_cols:
        feature_cols = _group_stat_feature_names(group_col, value_col=value_col)
        stats_df = train_df.groupBy(group_col).agg(
            count("*").alias(feature_cols[0]),
            avg(value_col).alias(feature_cols[1]),
            stddev(value_col).alias(feature_cols[2]),
        )
        stats_df = stats_df.fillna({name: 0.0 for name in feature_cols})
        stats_df.write.mode("overwrite").parquet((stats_root / group_col).as_posix())

        train_df = train_df.join(broadcast(stats_df), on=group_col, how="left")
        valid_df = valid_df.join(broadcast(stats_df), on=group_col, how="left")
        fill_map = {name: 0.0 for name in feature_cols}
        train_df = train_df.fillna(fill_map)
        valid_df = valid_df.fillna(fill_map)
        created_feature_cols.extend(feature_cols)

    train_df = add_group_behavior_features(train_df, value_col=value_col)
    valid_df = add_group_behavior_features(valid_df, value_col=value_col)
    created_feature_cols.extend(
        [
            feature_name
            for feature_name in [
                "log1p_amt_over_card1_mean",
            ]
            if feature_name in train_df.columns
        ]
    )

    return train_df, valid_df, available_group_cols, created_feature_cols


def apply_group_aggregation_maps(df, spark, stats_root, group_cols=None):
    stats_root = Path(stats_root)

    group_cols = group_cols or GROUP_STATS_DEFAULT_GROUPS
    if "uid_simple" in group_cols:
        df = add_uid_simple(df)
    available_group_cols = [group_col for group_col in group_cols if group_col in df.columns]

    for group_col in available_group_cols:
        stats_path = stats_root / group_col
        if stats_path.exists():
            stats_df = read_parquet_map(spark, stats_path)
            fill_map = {name: 0.0 for name in stats_df.columns if name != group_col}
            df = df.join(broadcast(stats_df), on=group_col, how="left")
            df = df.fillna(fill_map)
    return add_group_behavior_features(df)


# =========================
# scaling
# =========================

RAW_MODEL_EXCLUSION_COLUMNS = {"TransactionDT"}

def detect_continuous_columns(df, seuil_unique=10):
    """
    Détecte les colonnes numériques continues sans lancer un job Spark par colonne.

    Ancienne version :
    - faisait `distinct().count()` colonne par colonne
    - devenait très lente sur un dataset large avec beaucoup de variables

    Nouvelle version :
    - calcule une estimation du nombre de valeurs distinctes pour toutes les
      colonnes numériques en une seule agrégation Spark
    - garde la même logique métier : on ne scale que les colonnes ayant plus
      de `seuil_unique` modalités
    """
    numeric_types = {"double", "int", "bigint", "float", "smallint", "tinyint", "long", "short"}
    numeric_cols = [field.name for field in df.schema.fields if field.dataType.simpleString() in numeric_types]

    if not numeric_cols:
        return []

    approx_exprs = [approx_count_distinct(col_name).alias(col_name) for col_name in numeric_cols]
    distinct_counts = df.agg(*approx_exprs).first().asDict()

    continuous_cols = [col_name for col_name, unique_count in distinct_counts.items() if unique_count and unique_count > seuil_unique]
    return continuous_cols


def fit_and_apply_scaler(df, seuil_unique=10):
    exclusions = ["isFraud", *sorted(RAW_MODEL_EXCLUSION_COLUMNS)] + [
        c for c in df.columns if "is_" in c or "has_" in c or "OneHot" in c or "index" in c
    ]
    cols_to_scale = [c for c in detect_continuous_columns(df, seuil_unique=seuil_unique) if c not in exclusions]
    assembler = VectorAssembler(inputCols=cols_to_scale, outputCol="features_to_scale")
    df = assembler.transform(df)
    scaler = StandardScaler(
        inputCol="features_to_scale",
        outputCol="scaled_features",
        withMean=True,
        withStd=True,
    )
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    return df, scaler_model, cols_to_scale


def fit_and_apply_scaler_train_valid(train_df, valid_df, cols_to_scale=None, seuil_unique=10):
    """
    Fit le scaler uniquement sur train_df puis applique la même transformation
    au train et au validation set pour éviter le leakage.
    """
    cols_to_scale = cols_to_scale or detect_continuous_columns(train_df, seuil_unique=seuil_unique)
    exclusions = ["isFraud", *sorted(RAW_MODEL_EXCLUSION_COLUMNS)] + [
        c for c in train_df.columns if "is_" in c or "has_" in c or "OneHot" in c or "index" in c
    ]
    cols_to_scale = [c for c in cols_to_scale if c not in exclusions]

    assembler = VectorAssembler(inputCols=cols_to_scale, outputCol="features_to_scale")
    train_df = assembler.transform(train_df)
    valid_df = assembler.transform(valid_df)

    scaler = StandardScaler(
        inputCol="features_to_scale",
        outputCol="scaled_features",
        withMean=True,
        withStd=True,
    )
    scaler_model = scaler.fit(train_df)
    train_df = scaler_model.transform(train_df)
    valid_df = scaler_model.transform(valid_df)
    return train_df, valid_df, scaler_model, cols_to_scale


def apply_scaling_test(df, scaler_model, cols_to_scale):
    cols_to_scale = [colname for colname in cols_to_scale if colname in df.columns]
    if cols_to_scale:
        for colname in cols_to_scale:
            df = df.withColumn(colname, col(colname).cast("double"))
        # Certaines statistiques apprises au train n'ont pas de correspondance sur
        # des clés inédites du test. On neutralise ces nulls avant le scaling.
        df = df.fillna({colname: 0.0 for colname in cols_to_scale})

    assembler = VectorAssembler(
        inputCols=cols_to_scale,
        outputCol="features_to_scale",
        handleInvalid="keep",
    )
    df = assembler.transform(df)
    return scaler_model.transform(df)


# =========================
# encoding
# =========================

def fit_and_apply_encoding(df, vars_to_encode=None):
    vars_to_encode = vars_to_encode or VARS_TO_ENCODE
    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_idx", handleInvalid="keep")
        for column in vars_to_encode
    ]
    encoders = [OneHotEncoder(inputCol=f"{column}_idx", outputCol=f"{column}_vec") for column in vars_to_encode]
    pipeline = Pipeline(stages=indexers + encoders)
    model = pipeline.fit(df)
    return model.transform(df), model


def fit_and_apply_encoding_train_valid(train_df, valid_df, vars_to_encode=None):
    """
    Fit le pipeline d'encodage sur train_df uniquement, puis transforme train et valid.
    """
    vars_to_encode = vars_to_encode or [c for c in VARS_TO_ENCODE if c in train_df.columns]
    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_idx", handleInvalid="keep")
        for column in vars_to_encode
    ]
    encoders = [OneHotEncoder(inputCol=f"{column}_idx", outputCol=f"{column}_vec") for column in vars_to_encode]
    pipeline = Pipeline(stages=indexers + encoders)
    model = pipeline.fit(train_df)
    train_df = model.transform(train_df)
    valid_df = model.transform(valid_df)
    return train_df, valid_df, model


def apply_onehot_encoding_test(df, encoding_pipeline_model):
    return encoding_pipeline_model.transform(df)


def apply_frequency_encoding(df, frequency_root, columns=None):
    columns = columns or VARS_TO_FREQUENCY_ENCODE
    frequency_root = Path(frequency_root)
    ensure_dirs(frequency_root)
    total_count = df.count()

    for colname in columns:
        if colname not in df.columns:
            continue
        freq_df = df.groupBy(colname).agg(count("*").alias("cnt"))
        freq_df = freq_df.withColumn(f"{colname}_freq", col("cnt") / lit(total_count)).drop("cnt")
        freq_df.write.mode("overwrite").parquet((frequency_root / colname).as_posix())
        df = df.join(freq_df, on=colname, how="left")
    return df


def fit_and_apply_frequency_train_valid(train_df, valid_df, frequency_root, columns=None):
    """
    Calcule les fréquences sur train_df uniquement puis applique les mêmes maps sur valid_df.

    Version optimisée :
    - matérialise d'abord une source étroite avec uniquement les colonnes à encoder ;
    - calcule les maps depuis cette source stable ;
    - relit chaque map parquet avant les joins pour éviter de rejouer le groupBy.

    Sans cette coupure de lineage, Spark peut recalculer toute la chaîne
    scaling/onehot à chaque colonne, ce qui devient très lent et instable.
    """
    columns = [c for c in (columns or VARS_TO_FREQUENCY_ENCODE) if c in train_df.columns]
    frequency_root = Path(frequency_root)
    ensure_dirs(frequency_root)
    if not columns:
        return train_df, valid_df, columns

    spark = train_df.sparkSession
    source_path = frequency_root / "_train_frequency_source"

    print(f"Frequency encoding optimisé : matérialisation source étroite pour {len(columns)} colonnes")
    train_df.select(*columns).write.mode("overwrite").parquet(source_path.as_posix())
    train_freq_source = spark.read.parquet(source_path.as_posix())
    total_count = train_freq_source.count()

    for colname in columns:
        print(f"Frequency encoding : {colname}")
        freq_path = frequency_root / colname

        freq_df = train_freq_source.groupBy(colname).agg(count("*").alias("cnt"))
        freq_df = freq_df.withColumn(f"{colname}_freq", col("cnt") / lit(total_count)).drop("cnt")
        freq_df.write.mode("overwrite").parquet(freq_path.as_posix())

        freq_map_df = spark.read.parquet(freq_path.as_posix())
        train_df = train_df.join(broadcast(freq_map_df), on=colname, how="left")
        valid_df = valid_df.join(broadcast(freq_map_df), on=colname, how="left")
        train_df = train_df.fillna({f"{colname}_freq": 0.0})
        valid_df = valid_df.fillna({f"{colname}_freq": 0.0})
    return train_df, valid_df, columns


def apply_frequency_maps(df, spark, frequency_root, columns=None):
    columns = columns or VARS_TO_FREQUENCY_ENCODE
    frequency_root = Path(frequency_root)
    for colname in columns:
        freq_path = frequency_root / colname
        # Auto-create UID columns on-the-fly if absent (needed for inference on test data)
        if colname == "uid_simple" and "uid_simple" not in df.columns:
            df = add_uid_simple(df)
        if colname == "uid_full" and "uid_full" not in df.columns:
            df = add_uid_full(df)
        if colname in df.columns and freq_path.exists():
            freq_df = read_parquet_map(spark, freq_path)
            df = df.join(freq_df, on=colname, how="left")
            df = df.fillna({f"{colname}_freq": 0.0})
    return df


def apply_target_encoding(df, target_root, columns=None, target_col="isFraud"):
    columns = columns or VARS_TO_TARGET_ENCODE
    target_root = Path(target_root)
    ensure_dirs(target_root)
    fill_values = {}

    for colname in columns:
        if colname not in df.columns:
            continue
        encoded_col = f"{colname}_target_enc"
        target_df = df.groupBy(colname).agg(mean(target_col).alias(encoded_col))
        target_df.write.mode("overwrite").parquet((target_root / colname).as_posix())
        df = df.join(target_df, on=colname, how="left")
        global_mean = float(df.select(avg(target_col)).first()[0])
        df = df.fillna({encoded_col: global_mean})
        fill_values[encoded_col] = global_mean

    return df, fill_values


def apply_target_maps(df, spark, target_root, fill_values=None, columns=None):
    columns = columns or VARS_TO_TARGET_ENCODE
    fill_values = fill_values or {}
    target_root = Path(target_root)

    for colname in columns:
        encoded_col = f"{colname}_target_enc"
        target_path = target_root / colname
        if colname in df.columns and target_path.exists():
            target_df = read_parquet_map(spark, target_path)
            df = df.join(target_df, on=colname, how="left")
            default_value = fill_values.get(encoded_col, 0.0)
            df = df.fillna({encoded_col: default_value})
    return df


def fit_and_apply_target_train_valid(train_df, valid_df, target_root, columns=None, target_col="isFraud"):
    """
    Apprend les moyennes de target encoding sur train_df uniquement puis les applique à valid_df.
    """
    columns = [c for c in (columns or VARS_TO_TARGET_ENCODE) if c in train_df.columns]
    target_root = Path(target_root)
    ensure_dirs(target_root)
    fill_values = {}
    global_mean = float(train_df.select(avg(target_col)).first()[0])

    for colname in columns:
        encoded_col = f"{colname}_target_enc"
        target_df = train_df.groupBy(colname).agg(mean(target_col).alias(encoded_col))
        target_df.write.mode("overwrite").parquet((target_root / colname).as_posix())
        train_df = train_df.join(broadcast(target_df), on=colname, how="left")
        valid_df = valid_df.join(broadcast(target_df), on=colname, how="left")
        train_df = train_df.fillna({encoded_col: global_mean})
        valid_df = valid_df.fillna({encoded_col: global_mean})
        fill_values[encoded_col] = global_mean

    return train_df, valid_df, fill_values, columns


# =========================
# feature assembly
# =========================

def get_baseline_feature_columns(df):
    return [
        c
        for c in df.columns
        if c.endswith("_vec")
        or c.endswith("_freq")
        or c.endswith("_count")
        or c.endswith("_mean")
        or c.endswith("_std")
        or c.startswith("is_")
        or c.startswith("has_")
        or c.startswith("log1p_")
        or c.startswith("fp_")
        or (c.startswith("risk_") and not c.startswith("risk_key_"))
        or c == "scaled_features"
    ]


def assemble_final_features(df, final_features=None):
    final_features = final_features or get_baseline_feature_columns(df)
    if "features" in df.columns:
        df = df.drop("features")
    assembler = VectorAssembler(inputCols=final_features, outputCol="features")
    return assembler.transform(df)


def assemble_train_valid_features(train_df, valid_df, final_features=None):
    """
    Assemble le vecteur features sur train_df et valid_df avec la même liste de colonnes.
    """
    final_features = final_features or get_baseline_feature_columns(train_df)
    if "features" in train_df.columns:
        train_df = train_df.drop("features")
    if "features" in valid_df.columns:
        valid_df = valid_df.drop("features")

    assembler = VectorAssembler(inputCols=final_features, outputCol="features")
    train_df = assembler.transform(train_df)
    valid_df = assembler.transform(valid_df)
    return train_df, valid_df, final_features


def align_features_for_inference(df, final_features_list):
    missing_features = []
    for feature in final_features_list:
        if feature not in df.columns:
            if feature == "scaled_features":
                raise ValueError("La colonne scaled_features est absente du test.")
            df = df.withColumn(feature, lit(0.0))
            missing_features.append(feature)

    if "features" in df.columns:
        df = df.drop("features")

    # Le test peut contenir des valeurs inconnues/non vues au train. On neutralise
    # les nulls scalaires avant assemblage pour garder la dimension modele stable.
    schema = dict(df.dtypes)
    scalar_feature_columns = [
        feature
        for feature in final_features_list
        if schema.get(feature) in {"double", "float", "int", "bigint", "long", "smallint", "tinyint", "string"}
    ]
    if scalar_feature_columns:
        for feature in scalar_feature_columns:
            df = df.withColumn(feature, col(feature).cast("double"))
        df = df.fillna({feature: 0.0 for feature in scalar_feature_columns})

    assembler = VectorAssembler(inputCols=final_features_list, outputCol="features", handleInvalid="keep")
    return assembler.transform(df), missing_features


# =========================
# class weighting / metrics
# =========================

def add_class_weight(df, label_col="isFraud", weight_col="classWeight"):
    stats = (
        df.select(label_col)
        .agg(
            count("*").alias("total"),
            spark_sum(when(col(label_col) == 1, 1).otherwise(0)).alias("frauds"),
        )
        .first()
    )
    total = int(stats["total"])
    nb_fraudes = int(stats["frauds"] or 0)
    nb_nonfraudes = total - nb_fraudes
    if nb_fraudes == 0:
        raise ValueError("Aucune fraude trouvée dans le DataFrame.")
    fraud_weight = nb_nonfraudes / nb_fraudes
    return df.withColumn(weight_col, when(col(label_col) == 1, fraud_weight).otherwise(1.0))


def add_class_weight_with_train_ratio(train_df, valid_df=None, label_col="isFraud", weight_col="classWeight"):
    """
    Calcule le ratio de classes une seule fois sur le train,
    puis applique le même poids au train et au valid.
    """
    stats = (
        train_df.select(label_col)
        .agg(
            count("*").alias("total"),
            spark_sum(when(col(label_col) == 1, 1).otherwise(0)).alias("frauds"),
        )
        .first()
    )
    total = int(stats["total"])
    nb_fraudes = int(stats["frauds"] or 0)
    nb_nonfraudes = total - nb_fraudes
    if nb_fraudes == 0:
        raise ValueError("Aucune fraude trouvée dans le DataFrame d'entraînement.")

    fraud_weight = nb_nonfraudes / nb_fraudes
    train_df = train_df.withColumn(weight_col, when(col(label_col) == 1, fraud_weight).otherwise(1.0))

    if valid_df is not None:
        valid_df = valid_df.withColumn(weight_col, when(col(label_col) == 1, fraud_weight).otherwise(1.0))
        return train_df, valid_df, fraud_weight

    return train_df, fraud_weight


def dataframe_profile(df):
    return {
        "rows": df.count(),
        "columns": len(df.columns),
        "column_names": df.columns,
    }


def get_vector_size(df, colname):
    vector = df.select(colname).head()[0]
    if isinstance(vector, (DenseVector, SparseVector)):
        return vector.size
    return len(vector)



def metrics_at_threshold(scores, labels, threshold):
    """
    Calcule les métriques binaires principales pour un seuil donné.
    """
    import numpy as np

    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)
    pred = (scores > float(threshold)).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    n_labels = len(labels)
    accuracy = (tp + tn) / (n_labels if n_labels > 0 else 1)
    return {
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


# =========================
# notebook 07 helpers
# =========================

def load_three_way_blend_models(
    models_stage_dir,
    base_model_name="gbt_final_model",
    recall_model_names=None,
):
    """
    Recharge depuis le disque le GBT de base et les variantes classWeight.
    """
    from pyspark.ml.classification import GBTClassificationModel

    models_stage_dir = Path(models_stage_dir)
    recall_model_names = recall_model_names or ["gbt_classWeight_x1.15", "gbt_classWeight_x1.30"]

    base_path = (models_stage_dir / base_model_name).as_posix()
    if not Path(base_path).exists():
        raise ValueError(f"{base_model_name} absent sur disque ({base_path}).")

    base_model = GBTClassificationModel.load(base_path)
    loaded_variants = {}
    variant_paths = {}

    for model_name in recall_model_names:
        model_path = (models_stage_dir / model_name).as_posix()
        if not Path(model_path).exists():
            raise ValueError(f"Modèle {model_name} absent sur disque : {model_path}.")
        loaded_variants[model_name] = GBTClassificationModel.load(model_path)
        variant_paths[model_name] = model_path

    return base_model, loaded_variants, {
        "base_path": base_path,
        "variant_paths": variant_paths,
    }


def score_three_way_blend_models(valid_df, base_model, loaded_variants, safe_to_pandas_func):
    """
    Score les trois GBT et retourne un DataFrame pandas avec p_base/p_x115/p_x130.
    """
    from pyspark.ml.functions import vector_to_array

    def _score_and_rename(df, model, suffix):
        return (
            model.transform(df)
            .withColumnRenamed("rawPrediction", f"rawPrediction_{suffix}")
            .withColumnRenamed("probability", f"probability_{suffix}")
            .withColumnRenamed("prediction", f"prediction_{suffix}")
        )

    df_scored = _score_and_rename(valid_df, base_model, "b")
    df_scored = _score_and_rename(df_scored, loaded_variants["gbt_classWeight_x1.15"], "r15")
    df_scored = _score_and_rename(df_scored, loaded_variants["gbt_classWeight_x1.30"], "r30")

    scored_spark = (
        df_scored
        .withColumn("pb_arr", vector_to_array("probability_b"))
        .withColumn("pr15_arr", vector_to_array("probability_r15"))
        .withColumn("pr30_arr", vector_to_array("probability_r30"))
        .select(
            "isFraud",
            col("pb_arr")[1].alias("p_base"),
            col("pr15_arr")[1].alias("p_x115"),
            col("pr30_arr")[1].alias("p_x130"),
        )
    )
    scores3 = safe_to_pandas_func(scored_spark, context="blend max 3-way scores")
    scores3["isFraud"] = scores3["isFraud"].astype(int)
    return scores3



def score_three_way_blend_spark(df, base_model, loaded_variants, id_col=None, label_col=None):
    """
    Score les trois GBT et retourne uniquement les colonnes utiles au blend.
    Fonction pensée pour l'inférence: elle n'a besoin que de la colonne features.
    """
    from pyspark.ml.functions import vector_to_array

    def _score_and_rename(input_df, model, suffix):
        return (
            model.transform(input_df)
            .withColumnRenamed("rawPrediction", f"rawPrediction_{suffix}")
            .withColumnRenamed("probability", f"probability_{suffix}")
            .withColumnRenamed("prediction", f"prediction_{suffix}")
        )

    df_scored = _score_and_rename(df, base_model, "b")
    df_scored = _score_and_rename(df_scored, loaded_variants["gbt_classWeight_x1.15"], "r15")
    df_scored = _score_and_rename(df_scored, loaded_variants["gbt_classWeight_x1.30"], "r30")

    select_cols = []
    if id_col and id_col in df_scored.columns:
        select_cols.append(col(id_col))
    if label_col and label_col in df_scored.columns:
        select_cols.append(col(label_col))

    return (
        df_scored
        .withColumn("pb_arr", vector_to_array("probability_b"))
        .withColumn("pr15_arr", vector_to_array("probability_r15"))
        .withColumn("pr30_arr", vector_to_array("probability_r30"))
        .select(
            *select_cols,
            col("pb_arr")[1].alias("p_base"),
            col("pr15_arr")[1].alias("p_x115"),
            col("pr30_arr")[1].alias("p_x130"),
        )
    )


def fit_three_way_blend_calibrators(scores_pdf, random_state=42):
    """
    Fit les calibrateurs isotoniques du blend final sur la moitié calibration
    du split validation, comme dans le notebook 07.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split

    required_cols = {"isFraud", "p_base", "p_x115", "p_x130"}
    missing = required_cols.difference(scores_pdf.columns)
    if missing:
        raise ValueError(f"Colonnes absentes pour calibrer le blend: {sorted(missing)}")

    calib_pdf, _ = train_test_split(
        scores_pdf[list(required_cols)].copy(),
        test_size=0.5,
        stratify=scores_pdf["isFraud"],
        random_state=random_state,
    )
    labels = calib_pdf["isFraud"].astype(int).values

    calibrators = {}
    for score_col in ["p_base", "p_x115", "p_x130"]:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(calib_pdf[score_col].values, labels)
        calibrators[score_col] = iso
    return calibrators


def save_three_way_blend_calibrators(calibrators, path):
    """
    Sauvegarde les calibrateurs sklearn utilisés par le blend final.
    """
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrators, target.as_posix())
    return target.as_posix()


def load_three_way_blend_calibrators(path):
    """
    Recharge les calibrateurs du blend final.
    """
    import joblib

    target = Path(path)
    return joblib.load(target.as_posix())


def apply_three_way_blend_config(scores_pdf, blend_selection, calibrators=None):
    """
    Applique la config finale 3-way à un DataFrame pandas de scores p_base/p_x115/p_x130.
    Retourne les probabilités finales et la prédiction binaire.
    """
    scores_pdf = scores_pdf.copy()
    family = blend_selection.get("strategy", blend_selection.get("family", "raw"))

    if family == "cal":
        if calibrators is None:
            raise ValueError("Le blend final est calibré: fournir les calibrateurs isotoniques.")
        for score_col in ["p_base", "p_x115", "p_x130"]:
            scores_pdf[score_col + "_eff"] = calibrators[score_col].transform(scores_pdf[score_col].values)
    else:
        for score_col in ["p_base", "p_x115", "p_x130"]:
            scores_pdf[score_col + "_eff"] = scores_pdf[score_col]

    w_base = float(blend_selection["w_base"])
    w_15 = float(blend_selection["w_x1.15"])
    w_30 = float(blend_selection["w_x1.30"])
    threshold = float(blend_selection["threshold"])

    scores_pdf["fraud_probability"] = (
        w_base * scores_pdf["p_base_eff"]
        + w_15 * scores_pdf["p_x115_eff"]
        + w_30 * scores_pdf["p_x130_eff"]
    )
    scores_pdf["fraud_pred"] = (scores_pdf["fraud_probability"] >= threshold).astype(int)
    scores_pdf["threshold"] = threshold
    scores_pdf["blend_family"] = family
    return scores_pdf


def extract_feature_names_from_vector_metadata(df, features_col="features"):
    """
    Lit les noms détaillés des dimensions d'un vecteur Spark ML si la metadata existe.
    """
    metadata = df.schema[features_col].metadata
    ml_attr = metadata.get("ml_attr", {})
    num_attrs = int(ml_attr.get("num_attrs", 0))
    feature_names = [f"feature_{idx}" for idx in range(num_attrs)]

    for attrs in ml_attr.get("attrs", {}).values():
        for attr in attrs:
            idx = int(attr["idx"])
            if idx < len(feature_names):
                feature_names[idx] = attr.get("name", feature_names[idx])
    return feature_names


def model_feature_importance_frame(model, feature_names, top_n=30):
    """
    Convertit les importances Spark ML en DataFrame pandas lisible.
    """
    import pandas as pd

    importances = model.featureImportances.toArray()
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{idx}" for idx in range(len(importances))]

    importance_df = pd.DataFrame(
        {
            "feature_index": range(len(importances)),
            "feature": feature_names,
            "importance": importances,
        }
    )
    return importance_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def search_best_three_way_blend(
    scores3,
    grid_step=0.05,
    target_p=0.74,
    min_f1=0.72,
    min_p=0.73,
    max_p=0.75,
    min_r=0.70,
    fb_min_p=0.72,
    fb_min_r=0.70,
    random_state=42,
):
    """
    Recherche la meilleure combinaison 3-way sur les probabilités des 3 GBT.
    """
    import builtins
    import numpy as np
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split

    scores3 = scores3.copy()
    calib3, hold3 = train_test_split(
        scores3,
        test_size=0.5,
        stratify=scores3["isFraud"],
        random_state=random_state,
    )

    calib_labels = calib3["isFraud"].values
    for col_name in ["p_base", "p_x115", "p_x130"]:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(calib3[col_name].values, calib_labels)
        scores3[col_name + "_cal"] = iso.transform(scores3[col_name].values)

    calib3 = scores3.loc[calib3.index].copy()
    hold3 = scores3.loc[hold3.index].copy()

    weights = set()
    for a in np.round(np.arange(0.0, 1.0 + 1e-9, grid_step), 3):
        for b in np.round(np.arange(0.0, 1.0 - a + 1e-9, grid_step), 3):
            c = round(1.0 - a - b, 3)
            if c < -1e-9:
                continue
            weights.add((float(a), float(b), c if c > 0.0 else 0.0))
    weights = sorted(weights)
    fine_thresholds = np.round(np.arange(0.05, 0.95 + 1e-9, 0.002), 4)

    def _metrics_from_pred(labels, pred):
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(labels.sum() - tp)
        tn = int(len(labels) - tp - fp - fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1, precision, recall, tp, fp, fn, tn

    def _precision_gap(precision):
        if precision < min_p:
            return min_p - precision
        if precision > max_p:
            return precision - max_p
        return 0.0

    def _pick_calib(rows):
        ideal = [r for r in rows if r[1] >= min_f1 and min_p <= r[2] <= max_p and r[3] >= min_r]
        if ideal:
            return builtins.max(
                ideal,
                key=lambda r: (r[1], -builtins.abs(r[2] - target_p), r[3]),
            ), "ideal_window"

        hits = [r for r in rows if r[1] >= min_f1 and r[2] >= min_p and r[3] >= min_r]
        if hits:
            return builtins.max(
                hits,
                key=lambda r: (-_precision_gap(r[2]), r[1], -builtins.abs(r[2] - target_p), r[3]),
            ), "hit_F1_P_R"

        soft = [r for r in rows if r[2] >= fb_min_p and r[3] >= fb_min_r]
        if soft:
            return builtins.max(
                soft,
                key=lambda r: (r[1], -_precision_gap(r[2]), -builtins.abs(r[2] - target_p), r[3]),
            ), "soft_fallback"

        return builtins.max(rows, key=lambda r: r[1]), "raw_max_F1"

    best = None
    best_key = (-1, -1, -1.0, -1.0, -1.0, -1.0)
    audit_rows = []
    family_cols = {
        "raw": ("p_base", "p_x115", "p_x130"),
        "cal": ("p_base_cal", "p_x115_cal", "p_x130_cal"),
    }

    for family, (col_base, col_15, col_30) in family_cols.items():
        labels_cal = calib3["isFraud"].values
        c_base = calib3[col_base].values
        c_15 = calib3[col_15].values
        c_30 = calib3[col_30].values

        labels_hold = hold3["isFraud"].values
        h_base = hold3[col_base].values
        h_15 = hold3[col_15].values
        h_30 = hold3[col_30].values

        for w_base, w_15, w_30 in weights:
            score_cal = w_base * c_base + w_15 * c_15 + w_30 * c_30

            cal_eval_rows = []
            for threshold in fine_thresholds:
                pred = (score_cal >= threshold).astype(int)
                f1, precision, recall, tp, fp, fn, tn = _metrics_from_pred(labels_cal, pred)
                cal_eval_rows.append((threshold, f1, precision, recall))

            (best_cal_ts, best_cal_f1, best_cal_p, best_cal_r), cal_strat = _pick_calib(cal_eval_rows)

            score_hold = w_base * h_base + w_15 * h_15 + w_30 * h_30
            pred_hold = (score_hold >= best_cal_ts).astype(int)
            h_f1, h_p, h_r, h_tp, h_fp, h_fn, h_tn = _metrics_from_pred(labels_hold, pred_hold)

            ideal_pass = 1 if (h_f1 >= min_f1 and min_p <= h_p <= max_p and h_r >= min_r) else 0
            holdout_pass = 1 if (h_f1 >= min_f1 and h_p >= min_p and h_r >= min_r) else 0
            cmp_key = (ideal_pass, holdout_pass, h_f1, -_precision_gap(h_p), h_r, best_cal_f1)

            row = {
                "family": family,
                "w_base": w_base,
                "w_x1.15": w_15,
                "w_x1.30": w_30,
                "threshold": best_cal_ts,
                "cal_strat": cal_strat,
                "cal_F1": best_cal_f1,
                "cal_P": best_cal_p,
                "cal_R": best_cal_r,
                "hold_F1": h_f1,
                "hold_P": h_p,
                "hold_R": h_r,
                "hold_TP": h_tp,
                "hold_FP": h_fp,
                "hold_FN": h_fn,
                "hold_pass": holdout_pass,
                "ideal_pass": ideal_pass,
            }
            audit_rows.append(row)

            if cmp_key > best_key:
                best_key = cmp_key
                best = row

    all_blend_sweeps = pd.DataFrame(audit_rows)
    blend_config = {
        "selection_rule": "3-way blend max sweep",
        "strategy": best["family"],
        "w_base": best["w_base"],
        "w_x1.15": best["w_x1.15"],
        "w_x1.30": best["w_x1.30"],
        "threshold": best["threshold"],
        "metrics": {
            "holdout_Precision": best["hold_P"],
            "holdout_Recall": best["hold_R"],
            "holdout_F1": best["hold_F1"],
        },
    }

    return {
        "scores3": scores3,
        "calib3": calib3,
        "hold3": hold3,
        "weights_count": len(weights),
        "threshold_count": len(fine_thresholds),
        "all_blend_sweeps": all_blend_sweeps,
        "best": best,
        "blend_config": blend_config,
    }


def save_three_way_blend_metadata(best, metadata_dir, recall_selection_path):
    """
    Sauvegarde dans metadata la configuration complète du meilleur blend 3-way.
    """
    metadata_dir = Path(metadata_dir)
    recall_selection = load_json(recall_selection_path)
    eff_alpha = float(best["w_x1.15"]) + float(best["w_x1.30"])

    payload = {
        "selection_note": "3-way blend max sweep",
        "strategy": best["family"],
        "base_model": "gbt_final_model",
        "recall_model": recall_selection["selected_variant"],
        "recall_model_x1.15": "gbt_classWeight_x1.15",
        "recall_model_x1.30": "gbt_classWeight_x1.30",
        "blend_alpha_recall": eff_alpha,
        "w_base": float(best["w_base"]),
        "w_x1.15": float(best["w_x1.15"]),
        "w_x1.30": float(best["w_x1.30"]),
        "blend_formula": (
            f"proba = {best['w_base']:.2f}*proba_base"
            f" + {best['w_x1.15']:.2f}*proba_x1.15"
            f" + {best['w_x1.30']:.2f}*proba_x1.30"
        ),
        "threshold": float(best["threshold"]),
        "calibration_rule": best["cal_strat"],
        "holdout_F1": float(best["hold_F1"]),
        "holdout_Precision": float(best["hold_P"]),
        "holdout_Recall": float(best["hold_R"]),
    }
    save_json(metadata_dir / "blend_probability_candidate_selection.json", payload)
    return payload


def compute_three_way_blend_diagnostics(
    blend_selection,
    scores3=None,
    spark=None,
    models_stage_dir=None,
    valid_path=None,
    safe_to_pandas_func=None,
    random_state=42,
):
    """
    Reconstruit les scores finaux du blend 3-way et calcule les métriques finales.
    """
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    required_keys = {"w_base", "w_x1.15", "w_x1.30"}
    if not required_keys.issubset(blend_selection):
        raise ValueError("La config finale chargée n'est pas 3-way.")

    required_cols = {"isFraud", "p_base", "p_x115", "p_x130"}
    if scores3 is not None and required_cols.issubset(scores3.columns):
        final_blend_scores = scores3[["isFraud", "p_base", "p_x115", "p_x130"]].copy()
    else:
        if spark is None or models_stage_dir is None or valid_path is None or safe_to_pandas_func is None:
            raise ValueError("scores3 absent : fournir spark, models_stage_dir, valid_path et safe_to_pandas_func.")
        base_model, loaded_variants, _ = load_three_way_blend_models(models_stage_dir)
        valid_df = spark.read.parquet(Path(valid_path).as_posix())
        final_blend_scores = score_three_way_blend_models(
            valid_df,
            base_model,
            loaded_variants,
            safe_to_pandas_func,
        )

    final_blend_scores["isFraud"] = final_blend_scores["isFraud"].astype(int)
    calib_for_diag, _ = train_test_split(
        final_blend_scores[["isFraud", "p_base", "p_x115", "p_x130"]],
        test_size=0.5,
        stratify=final_blend_scores["isFraud"],
        random_state=random_state,
    )

    family = blend_selection.get("strategy", "raw")
    if family == "cal":
        calib_labels = calib_for_diag["isFraud"].values
        for src_col in ["p_base", "p_x115", "p_x130"]:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(calib_for_diag[src_col].values, calib_labels)
            final_blend_scores[src_col + "_eff"] = iso.transform(final_blend_scores[src_col].values)
    else:
        final_blend_scores["p_base_eff"] = final_blend_scores["p_base"]
        final_blend_scores["p_x115_eff"] = final_blend_scores["p_x115"]
        final_blend_scores["p_x130_eff"] = final_blend_scores["p_x130"]

    w_base = float(blend_selection["w_base"])
    w_15 = float(blend_selection["w_x1.15"])
    w_30 = float(blend_selection["w_x1.30"])
    threshold = float(blend_selection["threshold"])

    final_blend_scores["proba_blend"] = (
        w_base * final_blend_scores["p_base_eff"]
        + w_15 * final_blend_scores["p_x115_eff"]
        + w_30 * final_blend_scores["p_x130_eff"]
    )

    y_true = final_blend_scores["isFraud"].values.astype(int)
    y_score = final_blend_scores["proba_blend"].values
    final_metrics = metrics_at_threshold(y_score, y_true, threshold)
    final_metrics["AUC"] = roc_auc_score(y_true, y_score)
    final_metrics["AUC_PR"] = average_precision_score(y_true, y_score)
    final_metrics["threshold"] = threshold
    final_metrics["family"] = family
    final_metrics["w_base"] = w_base
    final_metrics["w_x1.15"] = w_15
    final_metrics["w_x1.30"] = w_30
    final_metrics["blend_alpha_recall"] = w_15 + w_30

    return final_blend_scores, pd.DataFrame([final_metrics])


# =========================
# notebook 08 helpers
# =========================

def prepare_test_dataset_for_stage08(spark, raw_data_dir, artifacts_root, train_optimized_stage_dir, compute_counts=True):
    """
    Rejoue toute la préparation du jeu de test à partir des artefacts
    appris dans les notebooks 03, 05 et 07.
    """
    artifacts_root = Path(artifacts_root)
    train_optimized_stage_dir = Path(train_optimized_stage_dir)

    artifacts_03 = artifacts_root / "03_quality_cleaning"
    artifacts_05 = artifacts_root / "05_feature_engineering"
    artifacts_07 = artifacts_root / "07_gbt_tuning_final_model"
    scaling_dir = artifacts_05 / "scaling"
    encoding_dir = artifacts_05 / "encoding"
    frequency_encoding_dir = encoding_dir / "frequency_encoding"
    target_encoding_dir = encoding_dir / "target_encoding"
    features_dir_05 = artifacts_05 / "features"
    metadata_dir_03 = artifacts_03 / "metadata"
    metadata_dir_07 = artifacts_07 / "metadata"
    fn_recovery_dir = artifacts_05 / "fn_recovery"

    cleaning_artifacts = load_json(metadata_dir_03 / "cleaning_artifacts.json")
    cols_to_drop = [col_name for col_name in cleaning_artifacts["cols_to_drop"] if col_name != "TransactionID"]
    cols_has_to_create = cleaning_artifacts["has_columns"]
    numeric_fill_values = cleaning_artifacts.get("numeric_fill_values", {})
    string_fill_columns = cleaning_artifacts.get("string_fill_columns", [])
    cols_to_scale = load_json(features_dir_05 / "cols_to_scale.json")
    frequency_columns = load_json(features_dir_05 / "frequency_columns.json")
    target_columns = load_json(features_dir_05 / "target_columns.json")
    final_features_list = load_json(metadata_dir_07 / "final_features_optimized.json")
    target_fill_values = load_json(artifacts_05 / "metadata" / "target_encoding_fill_values.json")

    scaler_model = StandardScalerModel.load((scaling_dir / "scaler_model").as_posix())
    encoding_pipeline_model = PipelineModel.load((encoding_dir / "onehot_pipeline_model").as_posix())

    test_df = load_test_merged(spark, raw_data_dir)
    test_df = clean_column_names(test_df)

    required_string_columns = set(VARS_TO_ENCODE + VARS_TO_FREQUENCY_ENCODE + [
        "DeviceType", "DeviceInfo", "P_emaildomain", "R_emaildomain",
        "ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    ])
    required_numeric_columns = set(cols_to_scale + cols_has_to_create + list(numeric_fill_values.keys()) + [
        "TransactionID", "TransactionDT", "TransactionAmt", "card1", "card2", "card3", "card5",
        "addr1", "addr2", "dist1", "C1", "D1", "D15",
    ])
    test_df = add_missing_inference_columns(
        test_df,
        numeric_columns=required_numeric_columns,
        string_columns=required_string_columns,
    )

    test_df = add_has_identity(test_df)
    test_df = clean_test_df(
        test_df,
        cols_to_drop=cols_to_drop,
        cols_has_to_create=cols_has_to_create,
        numeric_fill_values=numeric_fill_values,
        string_fill_columns=string_fill_columns,
    )
    test_df = add_baseline_features(test_df)
    test_df = add_log_features(test_df, cols_log=LOG_FEATURE_COLUMNS)
    test_df = add_post_model_features(test_df)
    test_df = add_recent_intense_feature(test_df, threshold=2.5)
    test_df = add_temporal_features(test_df)
    test_df = add_uid_simple(test_df)
    test_df = add_uid_full(test_df)
    test_df = add_m789_combo(test_df)
    test_df = apply_fn_recovery_maps(test_df, spark, fn_recovery_dir)
    test_df = apply_scaling_test(test_df, scaler_model, cols_to_scale)
    test_df = apply_onehot_encoding_test(test_df, encoding_pipeline_model)
    test_df = apply_frequency_maps(test_df, spark, frequency_encoding_dir, columns=frequency_columns)
    test_df = apply_target_maps(
        test_df,
        spark,
        target_encoding_dir,
        fill_values=target_fill_values,
        columns=target_columns,
    )
    test_df = add_false_positive_features(test_df)
    test_df, missing_features = align_features_for_inference(test_df, final_features_list)

    train_reference = spark.read.parquet((train_optimized_stage_dir / "train_features_optimized.parquet").as_posix())
    train_dim = get_vector_size(train_reference, "features") if compute_counts else None
    test_dim = get_vector_size(test_df, "features") if compute_counts else None

    summary = {
        "train_feature_dim": train_dim,
        "test_feature_dim": test_dim,
        "test_rows": int(test_df.count()) if compute_counts else None,
        "missing_features_filled_with_zero": missing_features,
        "fn_recovery_enabled_on_test": True,
        "group_stats_enabled_on_test": False,
    }

    context = {
        "cols_to_drop": cols_to_drop,
        "cols_has_to_create": cols_has_to_create,
        "cols_to_scale": cols_to_scale,
        "frequency_columns": frequency_columns,
        "target_columns": target_columns,
        "final_features_list": final_features_list,
    }

    return test_df, summary, context
