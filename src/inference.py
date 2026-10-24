from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import shutil

import pandas as pd


DEFAULT_KEY_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
]


@dataclass(frozen=True)
class FraudPrediction:
    transaction_id: int
    fraud_probability: float
    fraud_pred: int
    threshold: float
    blend_family: str
    risk_label: str


class FraudPredictionStore:
    """Fast lookup layer for the notebook 08 batch predictions.

    This is the right tool for the demo app: the user enters a TransactionID,
    and we return the already computed probability without starting Spark.
    """

    def __init__(
        self,
        project_dir: str | Path,
        predictions_path: str | Path | None = None,
        test_transaction_path: str | Path | None = None,
        test_identity_path: str | Path | None = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.predictions_path = Path(predictions_path) if predictions_path else (
            self.project_dir
            / "artifacts"
            / "08_test_preparation"
            / "predictions"
            / "test_predictions_blend.csv"
        )
        self.test_transaction_path = Path(test_transaction_path) if test_transaction_path else (
            self.project_dir / "data" / "test_transaction.csv"
        )
        self.test_identity_path = Path(test_identity_path) if test_identity_path else (
            self.project_dir / "data" / "test_identity.csv"
        )

        self._predictions: pd.DataFrame | None = None

    def load_predictions(self) -> pd.DataFrame:
        if self._predictions is None:
            if not self.predictions_path.exists():
                raise FileNotFoundError(
                    f"Predictions introuvables: {self.predictions_path}. "
                    "Lance d'abord le notebook 08."
                )
            predictions = pd.read_csv(self.predictions_path)
            if "TransactionID" not in predictions.columns:
                raise ValueError(f"Colonne TransactionID absente de {self.predictions_path}")
            self._predictions = predictions.set_index("TransactionID", drop=False)
        return self._predictions

    def get_prediction(self, transaction_id: int) -> FraudPrediction:
        predictions = self.load_predictions()
        if transaction_id not in predictions.index:
            raise KeyError(
                f"TransactionID {transaction_id} introuvable dans {self.predictions_path}. "
                "Pour une nouvelle transaction, il faudra passer par le scoring Spark/CSV."
            )

        row = predictions.loc[transaction_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        probability = float(row["fraud_probability"])
        threshold = float(row.get("threshold", 0.5))
        fraud_pred = int(row.get("fraud_pred", int(probability >= threshold)))

        return FraudPrediction(
            transaction_id=int(transaction_id),
            fraud_probability=probability,
            fraud_pred=fraud_pred,
            threshold=threshold,
            blend_family=str(row.get("blend_family", "unknown")),
            risk_label=self.risk_label(probability, threshold),
        )

    def get_predictions(self, transaction_ids: Iterable[int]) -> pd.DataFrame:
        rows = []
        missing = []
        for transaction_id in transaction_ids:
            try:
                prediction = self.get_prediction(int(transaction_id))
                rows.append(prediction.__dict__)
            except KeyError:
                missing.append(int(transaction_id))

        output = pd.DataFrame(rows)
        if missing:
            output.attrs["missing_transaction_ids"] = missing
        return output

    def get_transaction_details(self, transaction_id: int, columns: list[str] | None = None) -> dict:
        columns = columns or DEFAULT_KEY_COLUMNS
        details = self._find_row_by_transaction_id(
            self.test_transaction_path,
            transaction_id,
            columns=columns,
        )

        identity = self._find_row_by_transaction_id(
            self.test_identity_path,
            transaction_id,
            columns=["TransactionID", "DeviceType", "DeviceInfo", "id-30", "id-31"],
            missing_ok=True,
        )
        if identity:
            details.update({k: v for k, v in identity.items() if k != "TransactionID"})
        return details

    @staticmethod
    def risk_label(probability: float, threshold: float) -> str:
        if probability >= threshold:
            return "FRAUD"
        if probability >= threshold * 0.75:
            return "HIGH_RISK"
        if probability >= threshold * 0.35:
            return "MEDIUM_RISK"
        return "LOW_RISK"

    @staticmethod
    def _find_row_by_transaction_id(
        csv_path: Path,
        transaction_id: int,
        columns: list[str],
        chunksize: int = 100_000,
        missing_ok: bool = False,
    ) -> dict:
        if not csv_path.exists():
            if missing_ok:
                return {}
            raise FileNotFoundError(csv_path)

        available_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        usecols = [column for column in columns if column in available_columns]
        if "TransactionID" not in usecols:
            usecols.insert(0, "TransactionID")

        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
            match = chunk[chunk["TransactionID"] == transaction_id]
            if not match.empty:
                row = match.iloc[0].where(pd.notnull(match.iloc[0]), None)
                return row.to_dict()

        if missing_ok:
            return {}
        raise KeyError(f"TransactionID {transaction_id} introuvable dans {csv_path}")


def load_transaction_ids(path: str | Path) -> list[int]:
    ids_path = Path(path)
    if not ids_path.exists():
        raise FileNotFoundError(ids_path)

    if ids_path.suffix.lower() == ".txt":
        return [
            int(line.strip())
            for line in ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    df = pd.read_csv(ids_path)
    if "TransactionID" not in df.columns:
        raise ValueError("Le fichier doit contenir une colonne TransactionID.")
    return df["TransactionID"].dropna().astype(int).tolist()



def _copy_transaction_csv_for_spark(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _write_minimal_identity_csv(transaction_csv: Path, target_path: Path) -> None:
    """Create a TransactionID-only identity file when no identity CSV is supplied."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for chunk in pd.read_csv(transaction_csv, usecols=["TransactionID"], chunksize=100_000):
        chunk.to_csv(target_path, index=False, mode="w" if first else "a", header=first)
        first = False


def _safe_to_pandas(spark_df, spark, context="dataframe"):
    try:
        return spark_df.toPandas()
    except Exception as exc:
        print(f"[Spark->Pandas] Arrow a echoue pour {context}. Retry sans Arrow: {type(exc).__name__}")
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        pdf = spark_df.toPandas()
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        return pdf


def score_csv_with_spark(
    project_dir: str | Path,
    transaction_csv: str | Path,
    output_csv: str | Path,
    identity_csv: str | Path | None = None,
    app_name: str = "fraud_detection_csv_inference",
) -> tuple[pd.DataFrame, dict]:
    """Score a raw Kaggle-like transaction CSV with the saved Spark blend.

    The input must contain at least the transaction columns used by the project.
    Supplying the matching identity CSV gives the best fidelity; if omitted, the
    identity fields are treated as missing/unknown.
    """
    from pyspark.sql import SparkSession

    from src.fraud_pipeline import (
        apply_three_way_blend_config,
        fit_three_way_blend_calibrators,
        load_json,
        load_three_way_blend_calibrators,
        load_three_way_blend_models,
        prepare_test_dataset_for_stage08,
        save_three_way_blend_calibrators,
        score_three_way_blend_spark,
    )

    project_dir = Path(project_dir)
    transaction_csv = Path(transaction_csv)
    identity_csv = Path(identity_csv) if identity_csv else None
    output_csv = Path(output_csv)

    if not transaction_csv.exists():
        raise FileNotFoundError(transaction_csv)
    if identity_csv is not None and not identity_csv.exists():
        raise FileNotFoundError(identity_csv)

    staging_dir = project_dir / "artifacts" / "app_simulation" / "raw_upload"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_transaction = staging_dir / "test_transaction.csv"
    staged_identity = staging_dir / "test_identity.csv"

    _copy_transaction_csv_for_spark(transaction_csv, staged_transaction)
    if identity_csv is not None:
        shutil.copy2(identity_csv, staged_identity)
    else:
        _write_minimal_identity_csv(transaction_csv, staged_identity)

    artifacts_root = project_dir / "artifacts"
    train_stage_dir = artifacts_root / "07_gbt_tuning_final_model" / "data"
    models_stage_dir = project_dir / "models" / "07_gbt_tuning_final_model"
    metadata_dir = artifacts_root / "07_gbt_tuning_final_model" / "metadata"
    blend_config_path = metadata_dir / "blend_probability_candidate_selection.json"
    calibrators_path = metadata_dir / "blend_isotonic_calibrators.joblib"
    valid_split_path = train_stage_dir / "train_features_optimized_valid_split.parquet"

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[4]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        prepared_df, summary, _ = prepare_test_dataset_for_stage08(
            spark=spark,
            raw_data_dir=staging_dir,
            artifacts_root=artifacts_root,
            train_optimized_stage_dir=train_stage_dir,
            compute_counts=False,
        )
        scoring_cols = ["TransactionID", "features"]
        if "isFraud" in prepared_df.columns:
            scoring_cols.append("isFraud")

        prepared_path = project_dir / "artifacts" / "app_simulation" / "prepared_upload.parquet"
        prepared_df.select(*scoring_cols).write.mode("overwrite").parquet(prepared_path.as_posix())
        prepared_df = spark.read.parquet(prepared_path.as_posix()).cache()
        summary["prepared_rows"] = int(prepared_df.count())
        summary["prepared_parquet"] = prepared_path.as_posix()
        feature_row = prepared_df.select("features").first()
        if feature_row is not None:
            summary["test_feature_dim"] = int(feature_row["features"].size)

        blend_selection = load_json(blend_config_path)
        base_model, loaded_variants, model_paths = load_three_way_blend_models(models_stage_dir)
        summary["train_feature_dim"] = int(getattr(base_model, "numFeatures", summary.get("train_feature_dim") or 0))

        if calibrators_path.exists():
            calibrators = load_three_way_blend_calibrators(calibrators_path)
        else:
            valid_df = spark.read.parquet(valid_split_path.as_posix())
            valid_scores_spark = score_three_way_blend_spark(
                valid_df,
                base_model,
                loaded_variants,
                label_col="isFraud",
            )
            valid_scores_pdf = _safe_to_pandas(valid_scores_spark, spark, context="validation calibration")
            calibrators = fit_three_way_blend_calibrators(valid_scores_pdf, random_state=42)
            save_three_way_blend_calibrators(calibrators, calibrators_path)

        label_col = "isFraud" if "isFraud" in prepared_df.columns else None
        scores_spark = score_three_way_blend_spark(
            prepared_df,
            base_model,
            loaded_variants,
            id_col="TransactionID",
            label_col=label_col,
        )
        scores_pdf = _safe_to_pandas(scores_spark, spark, context="uploaded CSV scoring")
        predictions = apply_three_way_blend_config(scores_pdf, blend_selection, calibrators=calibrators)

        export_cols = [
            "TransactionID",
            "fraud_probability",
            "fraud_pred",
            "threshold",
            "blend_family",
            "p_base",
            "p_x115",
            "p_x130",
        ]
        if "isFraud" in predictions.columns:
            export_cols.insert(1, "isFraud")
        export_cols = [col for col in export_cols if col in predictions.columns]
        predictions_export = predictions[export_cols].sort_values("fraud_probability", ascending=False)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        predictions_export.to_csv(output_csv, index=False)

        summary.update(
            {
                "rows_scored": int(len(predictions_export)),
                "predicted_frauds": int(predictions_export["fraud_pred"].sum()),
                "predicted_fraud_rate": float(predictions_export["fraud_pred"].mean()) if len(predictions_export) else 0.0,
                "threshold": float(blend_selection["threshold"]),
                "blend_family": blend_selection.get("strategy", "raw"),
                "model_paths": model_paths,
                "output_csv": output_csv.as_posix(),
                "identity_csv_supplied": identity_csv is not None,
            }
        )
        summary_path = output_csv.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return predictions_export, summary
    finally:
        try:
            spark.stop()
        except Exception as exc:
            print(f"[Spark] stop warning: {type(exc).__name__}: {exc}")
