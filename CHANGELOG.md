# Changelog

## 0.4.0 — Monitoring and scientific evaluation

- Prediction logging, `/feedback`, `/stats`, confidence and top-k.
- Pinned reference dataset and Evidently drift tests.
- Grafana drift + API + model-performance dashboards.
- Webhook alerts and guarded retrain dispatch.
- Calibration (Brier/ECE), age/gender slices, 5-fold CV intervals, SHAP.
- Data card, runbook, architecture, and ADRs.

## 0.3.0 — CI/CD and infrastructure

- CI gates: lint, mypy, coverage, DVC repro, model quality, Docker smoke,
  Terraform validate, Checkov.
- MLflow `@champion` promotion and model metadata.
- Manual environment-gated deploy workflow and rollback script.
- AWS remote state, private ECS subnets, Secrets Manager, budget, bootstrap.

## 0.2.0 — Reproducible pipeline

- DVC owns data/model/reference outputs.
- Hydra env/model overrides fixed.
- Single sklearn pipeline with scaler inside.
- Label encoder fitted on train only; validation/test isolation.
- Data contract, ablation, model card.

## 0.1.0 — Initial project

- Hydra, Prefect, MLflow, DVC, FastAPI, Docker, Terraform, K8s.
