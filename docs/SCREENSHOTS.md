# Screenshot and Evidence Index

## Live product

| Image | Evidence |
|---|---|
| [`demo-predict.png`](../images/demo-predict.png) | Live `/demo` prediction with confidence and top-k |
| [`demo-feedback.png`](../images/demo-feedback.png) | Feedback capture with the request ID |
| [`monitoring.png`](../images/monitoring.png) | Live `/monitoring`: status, window, rows, coverage |
| [`monitoring-json.png`](../images/monitoring-json.png) | `/monitoring.json` machine-readable summary |
| [`swagger.png`](../images/swagger.png) | Swagger with the prefilled prediction example |
| [`swagger2.png`](../images/swagger2.png) | Alternate Swagger view |

## Operations

| Image | Evidence |
|---|---|
| [`cloud-run.png`](../images/cloud-run.png) | Cloud Run service and revision |
| [`cloudsql.png`](../images/cloudsql.png) | Cloud SQL backups and deletion protection |
| [`monitoring-job.png`](../images/monitoring-job.png) | Cloud Run Job execution history |
| [`scheduler.png`](../images/scheduler.png) | Daily 06:00 UTC scheduler |
| [`ci-green.png`](../images/ci-green.png) | GitHub Actions all green |
| [`mlflow.png`](../images/mlflow.png) | MLflow experiments / registry |
| [`dagshubexp.png`](../images/dagshubexp.png) | DagsHub experiment view |

## Local-only monitoring

The local compose stack still uses Grafana and Adminer. Those screenshots are
kept for the local monitoring runbook but are not part of the live GCP story:

- `images/grafana.png`
- `images/adminer.png`
- `images/prefect.png`

## Notes

- The live demo is GCP Cloud Run + Cloud SQL.
- AWS ECS Terraform is an optional reference stack and is not deployed.
- Do not commit screenshots containing secrets, tokens, database URLs, billing
  details, or personal email addresses.
