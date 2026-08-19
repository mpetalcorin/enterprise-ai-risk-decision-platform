#!/usr/bin/env bash
set -euo pipefail
python -m risk_platform.cli generate --rows 10000 --output data/transactions.csv
python -m risk_platform.cli train --data data/transactions.csv --backend xgboost --output models/risk_model.joblib
pytest -q
printf '\nStart the API with:\n  uvicorn risk_platform.api.main:app --host 0.0.0.0 --port 8000\n'
