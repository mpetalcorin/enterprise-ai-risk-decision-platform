# Model Card: Enterprise Risk Demonstration Model

## Intended use

This model is a **portfolio demonstration** of production ML engineering. It scores synthetic transaction-like records and routes high scores to `manual_review`. It is not validated for credit, fraud, AML, eligibility, customer treatment, or any real financial decision.

## Data

The training generator creates synthetic records with deliberately nonlinear relationships and an imbalanced positive class. No real customer, banking, clinical, or personal data are included.

## Inputs

Ten raw fields are transformed into twelve model features, including log transforms, transaction-to-baseline ratio, night-time indicator, and interaction terms. Validation rejects missing required fields, invalid binary values, nonpositive amounts, and out-of-range transaction hours.

## Candidate benchmark

One reproducible XGBoost run using 12,000 synthetic records, seed 42, produced:

| Metric | Held-out result |
|---|---:|
| Positive class prevalence | 5.43% |
| ROC-AUC | 0.885 |
| Average precision | 0.503 |
| Brier score | 0.075 |

These values are engineering smoke-test evidence only. They are not claims of financial-model performance.

## Explainability

The XGBoost backend uses SHAP TreeExplainer for local feature contributions. The PyTorch backend returns no explanation by default rather than attaching an explanation method that has not been validated in this reference implementation.

## Decision policy

The default threshold is 0.65. Scores at or above the threshold are routed to `manual_review`; other records receive `approve` in the demonstration API. A real system would require calibrated thresholds, policy/legal review, documented human override rules, fairness analysis, adverse-action controls where applicable, and independent validation.

## Monitoring

The service exposes prediction volume, latency, error, score and model-ready metrics. Training stores feature-distribution baselines for PSI drift checks. Drift detection does not substitute for outcome monitoring, calibration monitoring, subgroup analysis, bias/fairness analysis, data-quality monitoring, or periodic revalidation.

## Known limitations

- Synthetic labels and distributions do not reproduce a real financial process.
- No protected-characteristic features are generated; therefore the demonstration cannot establish fairness.
- The API-key mechanism is a minimal control, not enterprise identity and access management.
- The included PostgreSQL/Kubernetes resources are reference patterns, not a bank-approved platform blueprint.
- No automated model should make irreversible customer-impacting decisions solely from this demonstration score.
