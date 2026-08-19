from __future__ import annotations

import numpy as np
import pandas as pd

from risk_platform.features.pipeline import FEATURE_COLUMNS


class Explainer:
    def __init__(self, predictor):
        self.predictor = predictor
        self._explainer = None
        if predictor.backend == "xgboost":
            try:
                import shap

                self._explainer = shap.TreeExplainer(predictor.bundle["estimator"])
            except Exception:
                self._explainer = None

    def top_drivers(self, raw: pd.DataFrame, n: int = 5) -> list[list[dict[str, float | str]]]:
        features = self.predictor.feature_frame(raw)
        if self._explainer is None:
            return [[] for _ in range(len(features))]
        values = self._explainer.shap_values(features)
        if isinstance(values, list):
            values = values[-1]
        result: list[list[dict[str, float | str]]] = []
        for row in np.asarray(values):
            idx = np.argsort(np.abs(row))[::-1][:n]
            result.append(
                [
                    {
                        "feature": FEATURE_COLUMNS[i],
                        "contribution": float(row[i]),
                        "direction": "increases_risk" if row[i] >= 0 else "decreases_risk",
                    }
                    for i in idx
                ]
            )
        return result
