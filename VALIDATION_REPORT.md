# Validation Report

Validation performed on 19 August 2026 in the build environment.

## Executed successfully

- Python source compilation with `compileall`.
- TOML parsing for `pyproject.toml`.
- JSON parsing for the provisioned Grafana dashboard.
- YAML parsing for all Kubernetes, Prometheus, OpenTelemetry and Grafana provisioning files.
- Seven pytest tests covering data validation, feature engineering, prediction, drift report structure and the FastAPI real-time endpoint.
- XGBoost training on 12,000 synthetic records.
- Real-time API prediction with SHAP explanation.
- Batch inference over 12,000 records with batched audit persistence.
- Prometheus `/metrics` endpoint response.
- Model metadata endpoint response.
- Same-distribution PSI drift smoke test.
- Optional PyTorch training/prediction smoke test.

## XGBoost benchmark from the validated build

- synthetic positive prevalence: approximately 5.4%,
- held-out ROC-AUC: 0.885,
- held-out average precision: 0.503,
- held-out Brier score: 0.075.

These are synthetic engineering benchmarks, not estimates of performance on banking data.

## Not executed in this environment

Docker and Kubernetes command-line runtimes were not available in the build environment, so the Compose stack and a live Kubernetes cluster were not launched here. Their configuration files were syntax-parsed, and the application components they invoke were tested directly in Python. The GitHub Actions workflows are designed to perform a Docker image build in CI.

The MLflow server was not available in the build environment. Registry integration uses documented MLflow model aliases and artifact download APIs and is exercised when the Docker Compose stack or an external MLflow server is used.
