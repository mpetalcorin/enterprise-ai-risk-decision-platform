from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    model_backend: str = os.getenv("MODEL_BACKEND", "xgboost")
    model_path: str = os.getenv("MODEL_PATH", "models/risk_model.joblib")
    model_version: str = os.getenv("MODEL_VERSION", "local-dev")
    decision_threshold: float = float(os.getenv("DECISION_THRESHOLD", "0.65"))
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./risk_audit.db")
    require_api_key: bool = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
    api_key: str = os.getenv("API_KEY", "dev-only-key")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT", "enterprise-risk-platform")
    mlflow_registered_model: str = os.getenv("MLFLOW_REGISTERED_MODEL", "enterprise-risk-model")


settings = Settings()
