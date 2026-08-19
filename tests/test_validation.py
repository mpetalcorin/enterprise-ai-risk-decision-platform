import pandas as pd
import pytest

from risk_platform.data.synthetic import generate_synthetic_transactions
from risk_platform.data.validation import DataValidationError, validate_training_frame


def test_valid_synthetic_frame():
    frame = generate_synthetic_transactions(100)
    validate_training_frame(frame)


def test_missing_column_fails():
    frame = generate_synthetic_transactions(20).drop(columns=["transaction_amount"])
    with pytest.raises(DataValidationError):
        validate_training_frame(frame)


def test_invalid_binary_fails():
    frame = generate_synthetic_transactions(20)
    frame.loc[0, "device_new"] = 3
    with pytest.raises(DataValidationError):
        validate_training_frame(frame)
