# Non-Functional Requirements

This file makes operational expectations explicit so a proof of concept can be evaluated as a service rather than only as a model.

## Availability and resilience

| Requirement | Demonstration implementation | Production evidence needed |
|---|---|---|
| Liveness | `/health` | platform SLO monitoring |
| Readiness | `/ready` validates model runtime | dependency-aware readiness and synthetic probes |
| Replication | 3 Kubernetes API replicas | failure-domain distribution |
| Rollout | Kubernetes rolling update, zero configured unavailable | canary/blue-green where required |
| Autoscaling | HPA on CPU | workload-based scaling and capacity tests |
| Database resilience | local PostgreSQL reference | managed HA database, tested restore |

## Performance

Target values below are **acceptance targets, not measured claims**:

- single-record p95 inference latency: <150 ms under defined CPU load,
- batch API maximum: 5,000 records/request,
- no row-by-row database commits during batch scoring,
- bounded audit-write chunks,
- resource requests and limits defined for Kubernetes workloads.

Before production, run repeatable load tests and document p50/p95/p99 latency, throughput, saturation, database contention, cold-start behavior, and failure modes.

## Security

- API key supported for local/reference deployment.
- raw payloads are not written to the audit table.
- secrets are externalized from source configuration.
- container runs as non-root.
- production requires enterprise IAM, TLS/mTLS, network policy, key management, secret rotation, image signing, SBOM, vulnerability scanning, rate limiting and approved ingress controls.

## Observability

- structured JSON logging,
- Prometheus metrics,
- OpenTelemetry trace export when configured,
- Grafana/Tempo reference visualization,
- model/drift metrics and alert examples.

## Cost and operability

- CPU-first XGBoost default minimizes infrastructure requirements for the demo.
- optional PyTorch path demonstrates DL serving without making GPU a baseline dependency.
- horizontal scaling and model artifact separation support independent service lifecycle management.
- service runbook and model-promotion process define ownership actions.
