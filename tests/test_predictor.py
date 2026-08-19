from risk_platform.data.synthetic import generate_synthetic_transactions
from risk_platform.models.predictor import Predictor


def test_predictor_probabilities(trained_model):
    predictor = Predictor(trained_model)
    raw = generate_synthetic_transactions(10).drop(columns=["is_high_risk"])
    prob = predictor.predict_proba(raw)
    assert len(prob) == 10
    assert ((prob >= 0) & (prob <= 1)).all()
