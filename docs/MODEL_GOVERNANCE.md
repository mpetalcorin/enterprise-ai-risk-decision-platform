# Model Governance Workflow

## Lifecycle

```text
experiment
  → reproducible training run
  → packaged model + metrics + baseline
  → MLflow registered version
  → candidate alias
  → validation evidence
  → approval gate
  → champion alias
  → deployment promotion
  → monitoring
  → rollback / retirement
```

Training should register a **candidate**, not silently promote an unreviewed model into production. Promotion to `champion` should be an explicit controlled action after required evidence has been reviewed.

## Minimum promotion evidence

- source commit and container digest,
- training-data snapshot/version and provenance,
- exact feature contract,
- reproducibility seed/configuration,
- validation metrics and confidence intervals where appropriate,
- calibration analysis,
- class-imbalance strategy,
- subgroup/fairness analysis relevant to the use case,
- explainability review,
- data leakage review,
- security/privacy assessment,
- performance/load test,
- drift baseline,
- model limitations and intended-use statement,
- independent validation/approval record.

## Rollback conditions

Examples include severe service errors, materially degraded latency, failed data-quality controls, model/calibration degradation, significant unexplained drift, governance breach, or unexpected decision-rate change. Roll back the deployment/image and/or MLflow `champion` alias to the last approved model version, then preserve evidence for incident review.
