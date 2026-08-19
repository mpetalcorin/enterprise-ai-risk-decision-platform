from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly promote an approved MLflow model version")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--model-name", default="enterprise-risk-model")
    parser.add_argument("--version", required=True)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    print(f"Promoted {args.model_name} version {args.version} to alias '{args.alias}'")


if __name__ == "__main__":
    main()
