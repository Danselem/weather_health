# GCP Deployment

## Current live demo

| Item | Value |
|---|---|
| Project | `scenic-dynamo-447811-m9` |
| Region | `us-central1` |
| Cloud Run service | `weather-health` |
| URL | https://weather-health-587752925399.us-central1.run.app |
| Demo UI | https://weather-health-587752925399.us-central1.run.app/demo |
| Swagger | https://weather-health-587752925399.us-central1.run.app/docs |
| Image | `us-central1-docker.pkg.dev/scenic-dynamo-447811-m9/weather-health/weather-health:<timestamp>` |
| Model version | `1` (MLflow) |
| Database | Cloud SQL Postgres `weather-health-db` |
| Secret | `weather-health-db-url` |
| Max instances | `5` |
| Monitoring job | `weather-health-monitoring` |
| Scheduler | `weather-health-monitoring-daily` (06:00 UTC) |
| Terraform state | `gs://scenic-dynamo-447811-m9-tfstate` |

The live service connects to Cloud SQL through the Cloud Run connector and
stores predictions in the `weatherhealth` database. The database uses a
`db-f1-micro` instance, is zonal, and has automated backups disabled for the
demo. Monitor cost and enable backups before treating it as production.

## Deploy manually

Cloud Run requires `linux/amd64`. On Apple Silicon:

```bash
IMAGE=us-central1-docker.pkg.dev/$PROJECT_ID/weather-health/weather-health:$TAG
docker build --platform linux/amd64 \
  --build-arg MODEL_VERSION="$TAG" -t "$IMAGE" .
docker push "$IMAGE"

gcloud run deploy weather-health \
  --image "$IMAGE" \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 5 \
  --min-instances 0 \
  --add-cloudsql-instances scenic-dynamo-447811-m9:us-central1:weather-health-db \
  --set-env-vars MODEL_VERSION="$TAG",ENVIRONMENT=prod \
  --set-secrets PREDICTION_DB_URL=weather-health-db-url:latest
```

## Make targets

```bash
make gcp-bootstrap   # recreate Cloud SQL, secret, runtime SA, registry
make gcp-harden      # backups, deletion protection, budget alert
make gcp-monitoring-deploy  # scheduled Evidently Cloud Run Job
make gcp-monitoring-run     # execute once
make gcp-monitoring-status  # execution and scheduler status
make gcp-smoke       # health, ready, version, predict, feedback, stats
make gcp-status      # Cloud Run, Cloud SQL, secret, registry
make gcp-logs        # recent Cloud Run logs
make gcp-deploy      # build amd64 image + deploy
make gcp-deploy-current  # fetch champion, then build + deploy
make gcp-destroy CONFIRM=gcp-demo   # delete only the demo resources
make gcp-terraform-destroy          # Terraform-managed resources only
```

After a Terraform teardown, recreate infrastructure with `terraform apply`,
then run `make gcp-deploy`. `make gcp-bootstrap` is an emergency gcloud-only
recovery path, not the primary redeploy path.
`gcp-deploy` falls back to Cloud Build when Docker is unavailable and refuses
to run if Cloud SQL, the secret, or the runtime service account is missing.

`gcp-destroy` refuses to run unless `CONFIRM=gcp-demo` is set. It is
Terraform-aware: if state exists it disables SQL deletion protection and runs
`terraform destroy`; otherwise it falls back to targeted `gcloud` deletes.
It does not touch `elyra-*` services or unrelated IAM. `gcp-smoke`
resolves the live URL from Cloud Run, so it keeps working if the service URL
changes. `gcp-deploy` reads `MODEL_VERSION` from `models/model_metadata.json`
when available; set `FETCH_MODEL=1` or use `gcp-deploy-current` to fetch the
registered champion first.

## Smoke test

```bash
URL=https://weather-health-587752925399.us-central1.run.app
curl -s "$URL/health"
# open "$URL/demo" in a browser
curl -s "$URL/ready"
curl -s "$URL/version"
curl -s -X POST "$URL/predict" \
  -H 'Content-Type: application/json' --data @sample_input.json
```

Feedback and stats:

```bash
RID=$(curl -s -X POST "$URL/predict" \
  -H 'Content-Type: application/json' --data @sample_input.json \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

curl -s -X POST "$URL/feedback" \
  -H 'Content-Type: application/json' \
  -d "{\"request_id\":\"$RID\",\"true_label\":\"Heat Stroke\"}"
curl -s "$URL/stats"
```

Error paths:

```bash
jq 'del(.Age)' sample_input.json > /tmp/missing.json
curl -i -X POST "$URL/predict" \
  -H 'Content-Type: application/json' --data @/tmp/missing.json

curl -i -X POST "$URL/feedback" \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"does-not-exist","true_label":"Heat Stroke"}'
```

## Terraform path

`infra-gcp/` now models Cloud SQL, the Secret Manager secret, the runtime
service account, secret/Cloud SQL IAM, the connector annotation, and a private
GCS bucket. It still uses local state and has not been applied to the live
project.

Before using it as the source of truth:

- replace placeholder `project_id` values in `infra-gcp/environments/*`;
- add a GCS backend;
- import the existing live resources (see `infra-gcp/README.md`);
- decide whether Terraform or `gcloud run deploy` owns image updates.

## Scheduled monitoring

The monitoring job runs the Evidently flow with `--no-prefect` (Prefect's
engine is not needed in the job) against Cloud SQL. It writes
`monitoring_metrics`, and the job fails on drift so Cloud Scheduler can alert.

```bash
make gcp-monitoring-status
make gcp-monitoring-run
```

## Required permissions for a clean deploy

The current `terraform@` service account can create Artifact Registry
repositories, deploy Cloud Run, manage buckets, and create secrets. The Cloud
Run runtime service account needs `roles/cloudsql.client` and
`roles/secretmanager.secretAccessor`.

`infra-gcp/` now owns the live resources via a GCS backend and imports for
Cloud SQL, the secret, Artifact Registry, runtime service account, and Cloud
Run. `terraform plan` reports no changes. State is stored in
`gs://scenic-dynamo-447811-m9-tfstate` under `weather-health/prod`.
