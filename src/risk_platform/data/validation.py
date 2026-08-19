from __future__ import annotations

import pandas as pd

from risk_platform.data.synthetic import RAW_COLUMNS


class DataValidationError(ValueError):
    pass


def validate_training_frame(frame: pd.DataFrame, require_label: bool = True) -> None:
    required = set(RAW_COLUMNS)
    if require_label:
        required.add("is_high_risk")
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise DataValidationError("Dataset is empty")
    if frame[list(required)].isnull().any().any():
        raise DataValidationError("Null values detected in required fields")
    if (frame["transaction_amount"] <= 0).any() or (frame["avg_amount_30d"] <= 0).any():
        raise DataValidationError("Transaction and average amounts must be positive")
    for col in ("international", "high_risk_country", "device_new"):
        invalid = ~frame[col].isin([0, 1])
        if invalid.any():
            raise DataValidationError(f"{col} must contain only 0/1")
    if ((frame["transaction_hour"] < 0) | (frame["transaction_hour"] > 23)).any():
        raise DataValidationError("transaction_hour must be between 0 and 23")
    if require_label and not frame["is_high_risk"].isin([0, 1]).all():
        raise DataValidationError("is_high_risk must contain only 0/1")
