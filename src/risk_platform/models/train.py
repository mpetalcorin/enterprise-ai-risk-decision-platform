from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from risk_platform.data.validation import validate_training_frame
from risk_platform.features.pipeline import FEATURE_COLUMNS, engineer_features
from risk_platform.models.torch_model import RiskMLP


def _baseline_spec(features: pd.DataFrame, bins: int = 10) -> dict[str, Any]:
    spec: dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        values = features[col].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            edges = np.array([values.min() - 1e-9, values.max() + 1e-9])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf
        counts, _ = np.histogram(values, bins=edges)
        proportions = (counts / max(counts.sum(), 1)).tolist()
        spec[col] = {"edges": edges.tolist(), "proportions": proportions}
    return spec


def _version(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _train_xgboost(x_train: pd.DataFrame, y_train: pd.Series, seed: int) -> XGBClassifier:
    positives = max(int(y_train.sum()), 1)
    negatives = max(len(y_train) - positives, 1)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=negatives / positives,
        random_state=seed,
        n_jobs=4,
    )
    model.fit(x_train, y_train)
    return model


def _train_torch(x_train: pd.DataFrame, y_train: pd.Series, seed: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import torch

    torch.manual_seed(seed)
    mean = x_train.mean(axis=0).to_numpy(dtype=np.float32)
    std = x_train.std(axis=0).replace(0, 1).to_numpy(dtype=np.float32)
    x = ((x_train.to_numpy(dtype=np.float32) - mean) / std)
    y = y_train.to_numpy(dtype=np.float32)
    model = RiskMLP(x.shape[1])
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    model.train()
    for _ in range(120):
        optim.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optim.step()
    return model.state_dict(), mean, std


def train_model(
    frame: pd.DataFrame,
    backend: str = "xgboost",
    output_path: str | Path = "models/risk_model.joblib",
    threshold: float = 0.65,
    seed: int = 42,
) -> dict[str, Any]:
    validate_training_frame(frame, require_label=True)
    features = engineer_features(frame)
    target = frame["is_high_risk"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=seed
    )

    metadata = {
        "backend": backend,
        "feature_names": FEATURE_COLUMNS,
        "threshold": threshold,
        "created_unix": int(time.time()),
        "seed": seed,
        "positive_rate_train": float(y_train.mean()),
        "rows_train": len(x_train),
        "rows_test": len(x_test),
    }

    if backend == "xgboost":
        estimator = _train_xgboost(x_train, y_train, seed)
        prob = estimator.predict_proba(x_test)[:, 1]
        payload: dict[str, Any] = {"estimator": estimator}
    elif backend == "pytorch":
        import torch

        state_dict, mean, std = _train_torch(x_train, y_train, seed)
        model = RiskMLP(len(FEATURE_COLUMNS))
        model.load_state_dict(state_dict)
        model.eval()
        x_test_scaled = (x_test.to_numpy(dtype=np.float32) - mean) / std
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.tensor(x_test_scaled))).numpy()
        payload = {"state_dict": state_dict, "mean": mean, "std": std}
    else:
        raise ValueError("backend must be 'xgboost' or 'pytorch'")

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "average_precision": float(average_precision_score(y_test, prob)),
        "brier_score": float(brier_score_loss(y_test, prob)),
        "positive_rate_test": float(y_test.mean()),
    }
    metadata["metrics"] = metrics
    metadata["model_version"] = _version(metadata)

    bundle = {
        **payload,
        "metadata": metadata,
        "baseline": _baseline_spec(x_train),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return metadata
