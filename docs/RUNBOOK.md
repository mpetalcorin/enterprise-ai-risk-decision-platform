# Service Runbook

## Ownership

The service owner is responsible for availability, model artifact integrity, release coordination, audit continuity, monitoring response, documentation and escalation. Replace repository placeholders with the actual team/on-call rotation before deployment.

## Health triage

1. Check `/health` for process liveness.
2. Check `/ready` for model runtime readiness.
3. Check Prometheus `up{job="risk-api"}` and prediction error rate.
4. Check p95 latency and pod CPU/memory saturation.
5. Inspect structured logs using request/model version identifiers.
6. Inspect Tempo traces when OTLP tracing is enabled.
7. Confirm PostgreSQL availability and audit-write errors.
8. Check the active model version and recent deployment/image digest.

## Rollback

Kubernetes example:

```bash
kubectl -n enterprise-risk rollout history deployment/risk-api
kubectl -n enterprise-risk rollout undo deployment/risk-api
kubectl -n enterprise-risk rollout status deployment/risk-api
```

Model-registry rollback should re-point the approved `champion` alias to the last validated version and redeploy/reload according to the platform's release process.

## Drift response

- PSI <0.10: continue routine monitoring.
- PSI 0.10-0.25: investigate changed source distributions and data-quality changes.
- PSI >=0.25: treat as significant covariate drift in this demo, assess model performance/calibration before continued automated use.

Do not retrain automatically from drift alone. Retraining should enter the same validation and promotion process as any other model change.

## Audit failure

Prediction scoring and audit persistence are separated so a database failure does not silently alter a model score. In a real controlled environment, define whether audit-write failure must fail closed, queue events durably, or switch the service to degraded/manual-review mode. The correct behavior depends on business and regulatory requirements.
