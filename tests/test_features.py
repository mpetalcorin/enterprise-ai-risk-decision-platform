import numpy as np

from risk_platform.data.synthetic import generate_synthetic_transactions
from risk_platform.features.pipeline import FEATURE_COLUMNS, engineer_features


def test_feature_pipeline_is_finite():
    frame = generate_synthetic_transactions(100)
    features = engineer_features(frame)
    assert list(features.columns) == FEATURE_COLUMNS
    assert np.isfinite(features.to_numpy()).all()
