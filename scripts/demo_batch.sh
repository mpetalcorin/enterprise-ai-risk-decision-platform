#!/usr/bin/env bash
set -euo pipefail
python -m risk_platform.cli batch --input data/transactions.csv --output reports/batch_predictions.csv
head -n 5 reports/batch_predictions.csv
