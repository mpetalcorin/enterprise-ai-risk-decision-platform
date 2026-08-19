from __future__ import annotations

import time
import uuid
from pathlib import Path

import pandas as pd

from risk_platform.audit.repository import AuditRepository
from risk_platform.config import settings
from risk_platform.data.validation import validate_training_frame
from risk_platform.models.predictor import Predictor


def run_batch(input_path: str | Path, output_path: str | Path, model_path: str | Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    validate_training_frame(frame, require_label=False)
    predictor = Predictor(model_path or settings.model_path)
    raw = frame.drop(columns=["is_high_risk"], errors="ignore")
    started = time.perf_counter()
    probabilities = predictor.predict_proba(raw)
    elapsed = time.perf_counter() - started
    decisions = predictor.decisions(probabilities)
    out = frame.copy()
    out["risk_probability"] = probabilities
    out["risk_band"] = [predictor.risk_band(float(p)) for p in probabilities]
    out["decision"] = decisions
    out["model_version"] = predictor.version
    out["model_backend"] = predictor.backend
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    audit = AuditRepository(settings.database_url)
    per_row_ms = elapsed * 1000 / max(len(raw), 1)
    audit.write_many(
        [
            {
                "request_id": str(uuid.uuid4()),
                "model_version": predictor.version,
                "model_backend": predictor.backend,
                "record": record,
                "risk_probability": float(probability),
                "decision": decision,
                "latency_ms": per_row_ms,
            }
            for record, probability, decision in zip(
                raw.to_dict("records"), probabilities, decisions, strict=True
            )
        ]
    )
    return out
