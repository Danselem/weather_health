# ADR 0001: Single cloud target (AWS)

## Status
Accepted

## Context
The repository contained three overlapping Terraform paths: a legacy MLflow
EC2/RDS stack, AWS ECS, and GCP Cloud Run. Maintaining all three produced
inconsistent security and deployment stories.

## Decision
AWS ECS Fargate is the supported production path. `infra/` is archived and
`infra-gcp/` is an alternative reference only.

## Consequences
- One deployment workflow, one state backend, one security model.
- GCP remains useful as a demonstration but is not part of the CI/CD gate.
- Legacy resources are clearly marked and partially hardened.
