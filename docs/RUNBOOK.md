# Runbook

## Local development

```bash
uv sync --frozen
cp .env.example .env
dvc pull
make test
```

## Reproduce the pipeline

```bash
make dvc
# or
uv run dvc repro
```

Stages: `clean → validate → transform → train → evaluate → ablation → calibration →
slices → cross_validate → explain`.

## Evaluate and inspect

```bash
make check-metrics
make analysis
cat reports/evaluation_metrics.csv
```

## Serve locally

```bash
make serve_local
curl -s http://localhost:8080/health
curl -s http://localhost:8080/ready
curl -s -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' --data @sample_input.json
```

## Monitoring stack

```bash
make start-monitoring
make drift-demo
make observe
make stop-monitoring
```

Ports: API `8000`, Grafana `3000`, Adminer `8081`, Postgres `5432`.

## Common failures

| Symptom | Check | Fix |
|---|---|---|
| `dvc pull` 401 | `DAGSHUB_USER`, `DAGSHUB_TOKEN`, `.dvc/config.local` | Recreate DVC auth |
| `dvc pull` 404 | `dvc status --cloud` | `dvc push` from a machine with data |
| Hydra `MissingConfigException` | `config/env/*.yaml` present | Restore committed env configs |
| Coverage gate fails | `pytest --cov=src` | Add tests, do not lower the gate casually |
| Drift flow has no data | `PREDICTION_DB_URL` points at API store | Set worker env to the same Postgres URL |
| Retrain not triggered | `RETRAIN_*` env and failure streak | Check monitoring metrics and threshold |
| ECS task unhealthy | ALB target group `/health` | Check CloudWatch logs and task SG |

## Incident: drift alert

1. Read `reports/monitoring/alert_*.json` and the Grafana panels.
2. Classify: schema change, pipeline bug, training-serving skew, seasonality, or
   genuine population shift.
3. Roll back if a deploy coincides with onset.
4. Retrain only after ruling out upstream bugs.
5. Re-run the quality gate before promoting.

## Retrain

```bash
make retrain                 # local
# or
# GitHub Actions -> Retrain -> Run workflow
```

The retrain registers a new `@champion` only if the quality gate passes. It does
not deploy automatically.

## Deploy and rollback

```bash
# GitHub Actions -> Deploy -> choose dev/staging/prod
./scripts/rollback.sh <cluster> <service> <task-family> <revision>
```

## AWS state

```bash
make aws-bootstrap TF_STATE_BUCKET=... TF_LOCK_TABLE=... AWS_REGION=us-east-1
cp infra-aws/backend.hcl.example infra-aws/backend.hcl
make aws-init
make aws-plan
```

Never commit `backend.hcl`, tfstate, tfvars with secrets, or `.env`.
