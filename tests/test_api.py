import importlib

from fastapi.testclient import TestClient


def test_realtime_api(monkeypatch, trained_model):
    monkeypatch.setenv("MODEL_PATH", str(trained_model))
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("REQUIRE_API_KEY", "false")

    import risk_platform.config as config
    import risk_platform.api.main as main

    importlib.reload(config)
    main.settings = config.settings
    main._predictor = None
    main._explainer = None
    main._audit = None

    client = TestClient(main.app)
    payload = {
        "transaction": {
            "transaction_amount": 2500.0,
            "account_age_days": 30,
            "transactions_24h": 20,
            "avg_amount_30d": 120.0,
            "international": 1,
            "high_risk_country": 1,
            "device_new": 1,
            "failed_logins_24h": 3,
            "transaction_hour": 2,
            "customer_tenure_years": 0.1
        },
        "explain": False
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 <= body["risk_probability"] <= 1
    assert body["decision"] in {"approve", "manual_review"}
    assert body["model_version"]
