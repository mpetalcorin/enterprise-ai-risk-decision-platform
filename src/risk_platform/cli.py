from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from risk_platform.batch import run_batch
from risk_platform.config import settings
from risk_platform.data.synthetic import generate_synthetic_transactions
from risk_platform.data.validation import validate_training_frame
from risk_platform.features.pipeline import engineer_features
from risk_platform.models.predictor import Predictor
from risk_platform.models.registry import fetch_registered_bundle, register_with_mlflow
from risk_platform.models.train import train_model
from risk_platform.monitoring.drift import drift_report, write_report
from risk_platform.monitoring.metrics import DRIFT_PSI


def cmd_generate(args: argparse.Namespace) -> None:
    frame = generate_synthetic_transactions(args.rows, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {len(frame):,} synthetic rows to {path}")
    print(f"Positive rate: {frame['is_high_risk'].mean():.3%}")


def cmd_train(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.data)
    metadata = train_model(frame, backend=args.backend, output_path=args.output, threshold=args.threshold, seed=args.seed)
    print(json.dumps(metadata, indent=2))
    register_with_mlflow(
        args.output,
        metadata["metrics"],
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment,
        settings.mlflow_registered_model,
    )


def cmd_batch(args: argparse.Namespace) -> None:
    out = run_batch(args.input, args.output, args.model)
    print(f"Wrote {len(out):,} predictions to {args.output}")


def cmd_drift(args: argparse.Namespace) -> None:
    predictor = Predictor(args.model or settings.model_path)
    frame = pd.read_csv(args.input)
    validate_training_frame(frame, require_label=False)
    features = engineer_features(frame)
    report = drift_report(features, predictor.bundle["baseline"])
    DRIFT_PSI.set(report["max_psi"])
    write_report(report, args.output)
    print(json.dumps(report, indent=2))



def cmd_fetch_model(args: argparse.Namespace) -> None:
    path = fetch_registered_bundle(
        tracking_uri=args.tracking_uri or settings.mlflow_tracking_uri,
        registered_model_name=args.model_name or settings.mlflow_registered_model,
        alias=args.alias,
        output_path=args.output,
    )
    print(f"Downloaded approved model bundle to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="risk-platform")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="Generate synthetic transactional risk data")
    p.add_argument("--rows", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/transactions.csv")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train", help="Train and package a risk model")
    p.add_argument("--data", default="data/transactions.csv")
    p.add_argument("--backend", choices=["xgboost", "pytorch"], default="xgboost")
    p.add_argument("--output", default="models/risk_model.joblib")
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("batch", help="Run file-based batch inference")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("drift", help="Compare an inference dataset with the training baseline")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="reports/drift_report.json")
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("fetch-model", help="Fetch an approved model bundle from an MLflow alias")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--alias", default="champion")
    p.add_argument("--output", default="models/risk_model.joblib")
    p.set_defaults(func=cmd_fetch_model)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
