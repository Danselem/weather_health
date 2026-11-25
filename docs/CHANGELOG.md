# Changelog

## 1.0.1 — Monitoring guardrails and evidence

- Minimum-row guard: windows under `MONITORING_MIN_ROWS=300` report
  `insufficient_data` instead of drifting on thin samples.
- Drift gate uses dataset drift share (`DRIFT_SHARE_THRESHOLD=0.5`) rather than
  any single failing column.
- `/monitoring` shows the evaluation window, rows in window, minimum rows, and
  request coverage explicitly.
- `make gcp-seed` seeds demo traffic so the monitoring loop has labelled rows.
- Screenshot evidence index and live walkthrough video added.

## 1.0.0 — Live demo and observability

- Cloud Run + Cloud SQL live demo with persistent predictions.
- Terraform owns the live GCP stack; state in GCS.
- Cloud SQL backups, deletion protection, and a monthly budget.
- Scheduled Cloud Run monitoring job and daily Cloud Scheduler trigger.
- In-app `/monitoring` dashboard and JSON endpoint.
- Request-level coverage, retry, dead-letter, and optional strict logging.
- Optional API key, rate limiting, and request-size limits.
- Locust load-test profile; local benchmark p95 27 ms.
- Email uptime and drift alerts via Cloud Monitoring.


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
