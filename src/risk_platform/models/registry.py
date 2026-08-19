from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskBundlePyfunc:
    """Factory holder for an MLflow PythonModel implementation without hard import at module load."""

    @staticmethod
    def build():
        import mlflow.pyfunc

        class _RiskBundleModel(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                from risk_platform.models.predictor import Predictor

                self.predictor = Predictor(context.artifacts["bundle"])

            def predict(self, context, model_input, params=None):
                import pandas as pd

                probabilities = self.predictor.predict_proba(model_input)
                return pd.DataFrame(
                    {
                        "risk_probability": probabilities,
                        "risk_band": [self.predictor.risk_band(float(p)) for p in probabilities],
                        "decision": self.predictor.decisions(probabilities),
                        "model_version": self.predictor.version,
                    }
                )

        return _RiskBundleModel()


def register_with_mlflow(
    model_path: str | Path,
    metrics: dict[str, float],
    tracking_uri: str,
    experiment: str,
    registered_model_name: str,
) -> str | None:
    """Log a portable MLflow pyfunc model and register it as a candidate version."""
    if not tracking_uri:
        logger.info("MLflow registration skipped because MLFLOW_TRACKING_URI is empty")
        return None
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow is not installed; local model artifact remains available")
        return None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path), artifact_path="packaged_model")
        mlflow.pyfunc.log_model(
            name="model",
            python_model=RiskBundlePyfunc.build(),
            artifacts={"bundle": str(model_path)},
        )
        result = mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            registered_model_name,
        )
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(registered_model_name, "candidate", result.version)
        client.set_model_version_tag(registered_model_name, result.version, "approval_status", "candidate")
        return f"models:/{registered_model_name}/{result.version}"


def fetch_registered_bundle(
    *,
    tracking_uri: str,
    registered_model_name: str,
    alias: str,
    output_path: str | Path,
) -> Path:
    """Download the packaged joblib artifact from an approved MLflow model alias."""
    if not tracking_uri:
        raise ValueError("tracking_uri is required")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="risk-model-") as tmp:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{registered_model_name}@{alias}",
                dst_path=tmp,
                tracking_uri=tracking_uri,
            )
        )
        candidates = list(downloaded.rglob("*.joblib"))
        if not candidates:
            raise FileNotFoundError(f"No .joblib bundle found in MLflow model {registered_model_name}@{alias}")
        # The pyfunc artifact contains exactly one packaged bundle in this reference implementation.
        shutil.copy2(candidates[0], output)
    return output
