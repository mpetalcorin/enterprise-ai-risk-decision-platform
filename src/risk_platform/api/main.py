from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from risk_platform.api.dependencies import verify_api_key
from risk_platform.audit.repository import AuditRepository
from risk_platform.config import settings
from risk_platform.explainability.shap_explainer import Explainer
from risk_platform.logging_config import configure_logging
from risk_platform.models.predictor import Predictor
from risk_platform.monitoring.metrics import (
    MODEL_READY,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTIONS,
    RISK_SCORE,
)
from risk_platform.monitoring.tracing import configure_tracing
from risk_platform.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    Driver,
    PredictionRequest,
    PredictionResponse,
)

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise AI Risk Decision Platform",
    version="0.1.0",
    description="Governed reference service for real-time and batch ML risk decisions.",
)

configure_tracing(app)

_predictor: Predictor | None = None
_explainer: Explainer | None = None
_audit: AuditRepository | None = None


def _ensure_runtime() -> tuple[Predictor, Explainer, AuditRepository]:
    global _predictor, _explainer, _audit
    if _predictor is None:
        path = Path(settings.model_path)
        if not path.exists():
            MODEL_READY.set(0)
            raise HTTPException(status_code=503, detail=f"Model artifact not found: {path}")
        _predictor = Predictor(path)
        _explainer = Explainer(_predictor)
        MODEL_READY.set(1)
    if _audit is None:
        _audit = AuditRepository(settings.database_url)
    return _predictor, _explainer, _audit


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    predictor, _, _ = _ensure_runtime()
    return {"status": "ready", "model_version": predictor.version, "backend": predictor.backend}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _predict_records(records: list[dict], explain: bool) -> list[PredictionResponse]:
    predictor, explainer, audit = _ensure_runtime()
    raw = pd.DataFrame(records)
    started = time.perf_counter()
    try:
        probabilities = predictor.predict_proba(raw)
        drivers = explainer.top_drivers(raw) if explain else [[] for _ in records]
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failure")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    total_latency = time.perf_counter() - started
    per_record_ms = total_latency * 1000 / max(len(records), 1)
    responses: list[PredictionResponse] = []
    audit_rows: list[dict] = []
    for record, probability, row_drivers in zip(records, probabilities, drivers, strict=True):
        request_id = str(uuid.uuid4())
        decision = "manual_review" if probability >= predictor.threshold else "approve"
        PREDICTIONS.labels(decision, predictor.version, predictor.backend).inc()
        PREDICTION_LATENCY.observe(total_latency / max(len(records), 1))
        RISK_SCORE.observe(float(probability))
        audit_rows.append(
            {
                "request_id": request_id,
                "model_version": predictor.version,
                "model_backend": predictor.backend,
                "record": record,
                "risk_probability": float(probability),
                "decision": decision,
                "latency_ms": per_record_ms,
            }
        )
        responses.append(
            PredictionResponse(
                request_id=request_id,
                risk_probability=round(float(probability), 6),
                risk_band=predictor.risk_band(float(probability)),
                decision=decision,
                threshold=predictor.threshold,
                model_backend=predictor.backend,
                model_version=predictor.version,
                top_drivers=[Driver(**d) for d in row_drivers],
            )
        )
    try:
        audit.write_many(audit_rows)
    except Exception:
        logger.exception("Audit batch write failed")
    return responses


@app.post("/v1/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
def predict(request: PredictionRequest) -> PredictionResponse:
    return _predict_records([request.transaction.model_dump()], request.explain)[0]


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_api_key)])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = _predict_records([r.model_dump() for r in request.transactions], request.explain)
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)


@app.get("/v1/model", dependencies=[Depends(verify_api_key)])
def model_metadata() -> dict:
    predictor, _, _ = _ensure_runtime()
    return predictor.metadata
