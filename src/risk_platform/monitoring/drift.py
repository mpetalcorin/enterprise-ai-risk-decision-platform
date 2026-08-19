from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def population_stability_index(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    expected = np.clip(expected.astype(float), eps, None)
    actual = np.clip(actual.astype(float), eps, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(features: pd.DataFrame, baseline: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, dict[str, float | str]] = {}
    for col, spec in baseline.items():
        edges = np.asarray(spec["edges"], dtype=float)
        expected = np.asarray(spec["proportions"], dtype=float)
        counts, _ = np.histogram(features[col].to_numpy(dtype=float), bins=edges)
        actual = counts / max(counts.sum(), 1)
        psi = population_stability_index(expected, actual)
        status = "stable" if psi < 0.10 else "watch" if psi < 0.25 else "drift"
        rows[col] = {"psi": psi, "status": status}
    max_psi = max((float(v["psi"]) for v in rows.values()), default=0.0)
    overall = "stable" if max_psi < 0.10 else "watch" if max_psi < 0.25 else "drift"
    return {"overall_status": overall, "max_psi": max_psi, "features": rows}


def write_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
