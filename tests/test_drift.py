from risk_platform.data.synthetic import generate_synthetic_transactions
from risk_platform.features.pipeline import engineer_features
from risk_platform.models.predictor import Predictor
from risk_platform.monitoring.drift import drift_report


def test_drift_report_structure(trained_model):
    predictor = Predictor(trained_model)
    raw = generate_synthetic_transactions(500, seed=99).drop(columns=["is_high_risk"])
    report = drift_report(engineer_features(raw), predictor.bundle["baseline"])
    assert report["overall_status"] in {"stable", "watch", "drift"}
    assert report["max_psi"] >= 0
    assert report["features"]
