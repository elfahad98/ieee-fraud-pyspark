from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.inference import FraudPredictionStore, load_transaction_ids, score_csv_with_spark


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "artifacts" / "app_simulation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulation locale de detection de fraude par TransactionID."
    )
    parser.add_argument(
        "--transaction-id",
        type=int,
        help="TransactionID a chercher dans les predictions du notebook 08.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="CSV ou TXT contenant une liste de TransactionID a chercher dans les predictions deja calculees.",
    )
    parser.add_argument(
        "--score-csv",
        type=Path,
        help="CSV brut de transactions a scorer avec Spark et le modele final.",
    )
    parser.add_argument(
        "--identity-csv",
        type=Path,
        help="CSV identity associe au CSV de transactions. Optionnel, mais recommande.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "predictions_lookup.csv",
        help="Chemin de sortie pour le mode batch ID ou le scoring CSV.",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Sauvegarde un petit graphique local pour la transaction demandee.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Lance un prompt local pour saisir des TransactionID.",
    )
    return parser


def print_prediction(payload: dict) -> None:
    prediction = payload["prediction"]
    details = payload.get("details", {})

    print("\n=== Prediction fraude ===")
    print(f"TransactionID      : {prediction['transaction_id']}")
    print(f"Probabilite fraude : {prediction['fraud_probability']:.4f}")
    print(f"Seuil decision     : {prediction['threshold']:.4f}")
    print(f"Decision           : {'FRAUD' if prediction['fraud_pred'] == 1 else 'NON_FRAUD'}")
    print(f"Niveau risque      : {prediction['risk_label']}")
    print(f"Famille blend      : {prediction['blend_family']}")

    if details:
        print("\n--- Details transaction ---")
        for key, value in details.items():
            print(f"{key:18}: {value}")


def prediction_payload(store: FraudPredictionStore, transaction_id: int) -> dict:
    prediction = store.get_prediction(transaction_id)
    details = store.get_transaction_details(transaction_id)
    return {
        "prediction": prediction.__dict__,
        "details": details,
    }


def save_chart(payload: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = payload["prediction"]

    probability = float(prediction["fraud_probability"])
    threshold = float(prediction["threshold"])
    colors = ["#d62828" if probability >= threshold else "#2a9d8f", "#264653"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Probabilite fraude", "Seuil"], [probability, threshold], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Risque transaction {prediction['transaction_id']}")
    ax.axhline(threshold, color="#264653", linestyle="--", linewidth=1)

    for index, value in enumerate([probability, threshold]):
        ax.text(index, value + 0.02, f"{value:.3f}", ha="center", fontweight="bold")

    fig.tight_layout()
    chart_path = output_dir / f"transaction_{prediction['transaction_id']}_risk.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def run_single(store: FraudPredictionStore, transaction_id: int, chart: bool = False) -> None:
    payload = prediction_payload(store, transaction_id)
    print_prediction(payload)

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"transaction_{transaction_id}_prediction.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON sauvegarde : {json_path}")

    if chart:
        chart_path = save_chart(payload, output_dir=output_dir)
        print(f"Graphique sauvegarde : {chart_path}")


def run_batch(store: FraudPredictionStore, ids_file: Path, output: Path) -> None:
    transaction_ids = load_transaction_ids(ids_file)
    predictions = store.get_predictions(transaction_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)

    print(f"Transactions demandees : {len(transaction_ids)}")
    print(f"Transactions trouvees  : {len(predictions)}")
    missing = predictions.attrs.get("missing_transaction_ids", [])
    if missing:
        print(f"Transactions absentes  : {missing[:20]}")
    print(f"Sortie batch           : {output}")


def run_score_csv(transaction_csv: Path, identity_csv: Path | None, output: Path) -> None:
    predictions, summary = score_csv_with_spark(
        project_dir=PROJECT_DIR,
        transaction_csv=transaction_csv,
        identity_csv=identity_csv,
        output_csv=output,
    )

    print("\n=== Scoring CSV termine ===")
    print(f"Lignes scorees        : {summary['rows_scored']}")
    print(f"Fraudes predites     : {summary['predicted_frauds']}")
    print(f"Taux fraude predit   : {summary['predicted_fraud_rate']:.4%}")
    print(f"Seuil                : {summary['threshold']:.4f}")
    print(f"Sortie predictions   : {output}")
    print(f"Sortie resume        : {output.with_suffix('.summary.json')}")
    print("\nTop transactions risquees :")
    print(predictions.head(10).to_string(index=False))


def run_interactive(store: FraudPredictionStore, chart: bool = False) -> None:
    print("Mode interactif. Tape un TransactionID, ou 'q' pour quitter.")
    while True:
        value = input("\nTransactionID > ").strip()
        if value.lower() in {"q", "quit", "exit"}:
            break
        if not value:
            continue
        try:
            run_single(store, int(value), chart=chart)
        except Exception as exc:
            print(f"Erreur : {exc}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.score_csv is not None:
        output = args.output
        if output == DEFAULT_OUTPUT_DIR / "predictions_lookup.csv":
            output = DEFAULT_OUTPUT_DIR / "scored_uploaded_csv.csv"
        run_score_csv(args.score_csv, args.identity_csv, output)
        return

    store = FraudPredictionStore(PROJECT_DIR)

    if args.transaction_id is not None:
        run_single(store, args.transaction_id, chart=args.chart)
        return

    if args.ids_file is not None:
        run_batch(store, args.ids_file, args.output)
        return

    if args.interactive:
        run_interactive(store, chart=args.chart)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
