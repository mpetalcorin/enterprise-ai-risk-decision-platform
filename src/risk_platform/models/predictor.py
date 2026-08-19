from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from risk_platform.features.pipeline import FEATURE_COLUMNS, engineer_features
from risk_platform.models.torch_model import RiskMLP


class Predictor:
    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.bundle = joblib.load(self.model_path)
        self.metadata: dict[str, Any] = self.bundle["metadata"]
        self.backend = self.metadata["backend"]
        self.threshold = float(self.metadata["threshold"])
        self.version = self.metadata["model_version"]
        self._torch_model = None
        if self.backend == "pytorch":
            import torch

            model = RiskMLP(len(FEATURE_COLUMNS))
            model.load_state_dict(self.bundle["state_dict"])
            model.eval()
            self._torch_model = model
            self._torch = torch

    def feature_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(raw)

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        x = self.feature_frame(raw)
        if self.backend == "xgboost":
            return self.bundle["estimator"].predict_proba(x)[:, 1]
        mean = self.bundle["mean"]
        std = self.bundle["std"]
        xt = self._torch.tensor((x.to_numpy(dtype=np.float32) - mean) / std)
        with self._torch.no_grad():
            return self._torch.sigmoid(self._torch_model(xt)).numpy()

    def decisions(self, probabilities: np.ndarray) -> list[str]:
        return ["manual_review" if p >= self.threshold else "approve" for p in probabilities]

    @staticmethod
    def risk_band(probability: float) -> str:
        if probability < 0.30:
            return "low"
        if probability < 0.65:
            return "medium"
        return "high"
