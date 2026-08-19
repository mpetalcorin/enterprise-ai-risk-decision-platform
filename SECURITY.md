# Security Policy and Reference Controls

This repository is a portfolio reference implementation. Do not expose it to untrusted networks with example credentials.

## Included controls

- environment-based secrets,
- optional API-key enforcement,
- non-root application container,
- input validation and payload size constraints,
- no raw prediction payload persistence in the audit table,
- model and request identifiers in operational records,
- health/readiness endpoints,
- documented production-hardening requirements.

## Required before real deployment

Use organization-approved IAM, TLS/mTLS, secrets management/rotation, private networking, firewall/network policies, dependency and container vulnerability scanning, SBOMs, signed images, least-privilege database roles, encryption at rest, rate limiting/WAF controls, backup/restore, penetration testing, incident response and data-retention policy.

Never commit real API keys, database passwords, customer data or production model artifacts to the repository.
