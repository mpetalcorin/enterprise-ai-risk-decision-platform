from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS = Counter(
    "risk_platform_predictions_total", "Total predictions", ["decision", "model_version", "backend"]
)
PREDICTION_LATENCY = Histogram(
    "risk_platform_prediction_latency_seconds", "Prediction latency in seconds"
)
RISK_SCORE = Histogram(
    "risk_platform_risk_probability", "Predicted risk probability", buckets=(0.1,0.2,0.3,0.5,0.65,0.8,0.9,1.0)
)
PREDICTION_ERRORS = Counter("risk_platform_prediction_errors_total", "Prediction errors")
DRIFT_PSI = Gauge("risk_platform_max_feature_psi", "Maximum feature population stability index")
MODEL_READY = Gauge("risk_platform_model_ready", "Whether a model is loaded and ready")
