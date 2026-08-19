#!/usr/bin/env bash
set -euo pipefail
curl -sS -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${API_KEY:-change-me-before-production}" \
  -d '{"transaction":{"transaction_amount":2400,"account_age_days":38,"transactions_24h":17,"avg_amount_30d":145,"international":1,"high_risk_country":1,"device_new":1,"failed_logins_24h":3,"transaction_hour":2,"customer_tenure_years":0.1},"explain":true}' | python -m json.tool
