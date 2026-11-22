# Monitoring

This stack closes the production loop:

1. The API logs every prediction (features, label, confidence, model version,
   latency) to SQLite locally or PostgreSQL in the compose stack.
2. `src/evidently_metrics.py` compares the current window with the pinned
   reference (`data/processed/reference.parquet`) and writes drift metrics to
   PostgreSQL.
3. Grafana reads those metrics and shows drift, missing values, and failed
   tests.
4. A failed drift test writes an alert JSON under `reports/monitoring/` and can
   fail a pipeline with `--fail-on-drift`.

## Start

```bash
make start-monitoring
```

Services:

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Grafana | http://localhost:3000 (admin/admin) |
| Adminer | http://localhost:8081 |
| PostgreSQL | localhost:5432, database `test` |

Send traffic:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' --data @sample_input.json
```

Generate a drifted window and run the gate:

```bash
PREDICTION_DB_URL=postgresql://postgres:example@localhost:5432/test \
  uv run python -m src.simulate_drift
```

Compare live logged predictions with the reference:

```bash
PREDICTION_DB_URL=postgresql://postgres:example@localhost:5432/test \
  uv run python -m src.evidently_metrics --fail-on-drift
```

Reports land in `reports/monitoring/`; Grafana panels read
`monitoring_metrics` and `performance_metrics` from PostgreSQL.

## Schedule and alerts

Deploy the daily monitoring flow once a Prefect work pool (`weather`) is running:

```bash
make prefect-pool       # first time only
make monitoring-deploy

# The worker must see the same prediction store as the API:
export PREDICTION_DB_URL=postgresql://postgres:example@localhost:5432/test
make worker             # in another terminal
```

The deployment runs daily at 06:00 UTC. Its pull step runs `uv sync --frozen`
and `dvc pull`, so the worker needs DVC credentials available (for example via
`DAGSHUB_USER`/`DAGSHUB_TOKEN` and `.dvc/config.local`).

Set optional integrations:

```bash
export ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...
export RETRAIN_WEBHOOK_URL=https://example.com/retrain
# or trigger the GitHub Retrain workflow:
export RETRAIN_GITHUB_REPO=Danselem/weather_health
export RETRAIN_GITHUB_TOKEN=<PAT with repo scope>
export RETRAIN_FAILURE_THRESHOLD=2
```

A failed report sends an alert and, after `RETRAIN_FAILURE_THRESHOLD`
consecutive failures, dispatches `retrain`. The retrain workflow trains, evaluates,
runs the quality gate, and registers a new champion; it never deploys
automatically.

## Performance from feedback

```bash
make performance
```

This joins `/feedback` labels to predictions and writes accuracy/macro F1 to
`performance_metrics` for the Grafana panel.

## Live dashboard

The Cloud Run API serves a read-only monitoring page at `/monitoring` and
`/monitoring.json`. It reads `monitoring_metrics` and `performance_metrics`
from Cloud SQL.

## Scheduled GCP job

The live demo also runs the same Evidently flow as a Cloud Run Job:

```bash
make gcp-monitoring-deploy   # build image, create job + 06:00 UTC schedule
make gcp-monitoring-run      # execute once
make gcp-monitoring-status   # recent executions + scheduler state
```

The job uses `PREDICTION_DB_URL` from Secret Manager, the Cloud SQL
connector, and `--no-prefect` (the Prefect engine is not needed inside the
job). It exits non-zero on drift.

## Retention

```bash
make prune-predictions
```

## Delayed labels

Send ground truth when it arrives:

```bash
curl -s -X POST http://localhost:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"<id from /predict>","true_label":"Heat Stroke"}'
```

## Triage playbook

| Signal | Likely cause | Action |
|---|---|---|
| Many columns drift at once | Upstream schema/ETL change | Diff schema, fix pipeline, re-run report |
| One column drifts | Pipeline bug or encoding change | Check deploy timestamps, roll back |
| Prediction drift, features stable | Concept drift or label-pipeline issue | Inspect recent labels, consider retrain |
| Drift only in serving data | Training-serving skew | Compare feature path, regenerate reference |
| Calendar-aligned drift | Seasonality | Widen thresholds, document as known |
| Genuine population shift | New segment/behavior | Retrain on the new window, re-run quality gate |

Do not retrain before ruling out schema and pipeline bugs.

## Stop

```bash
cd monitoring && docker compose down
```
