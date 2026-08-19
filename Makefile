.PHONY: install test lint generate train serve batch drift docker-up docker-down

install:
	python -m pip install -e '.[dev]'

test:
	pytest --cov=risk_platform --cov-report=term-missing

lint:
	ruff check src tests

generate:
	python -m risk_platform.cli generate --rows 10000 --output data/transactions.csv

train:
	python -m risk_platform.cli train --data data/transactions.csv --backend xgboost --output models/risk_model.joblib

serve:
	uvicorn risk_platform.api.main:app --host 0.0.0.0 --port 8000 --reload

batch:
	python -m risk_platform.cli batch --input data/transactions.csv --output reports/batch_predictions.csv

drift:
	python -m risk_platform.cli drift --input data/transactions.csv --output reports/drift_report.json

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v
