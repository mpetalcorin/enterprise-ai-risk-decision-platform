from __future__ import annotations

from pathlib import Path

import pytest

from risk_platform.data.synthetic import generate_synthetic_transactions
from risk_platform.models.train import train_model


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("model") / "risk_model.joblib"
    frame = generate_synthetic_transactions(1800, seed=7)
    train_model(frame, backend="xgboost", output_path=path, seed=7)
    return path
