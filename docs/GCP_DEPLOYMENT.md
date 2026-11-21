# GCP Deployment

## Current live demo

| Item | Value |
|---|---|
| Project | `scenic-dynamo-447811-m9` |
| Region | `us-central1` |
| Cloud Run service | `weather-health` |
| URL | https://weather-health-587752925399.us-central1.run.app |
| Swagger | https://weather-health-587752925399.us-central1.run.app/docs |
| Image | `us-central1-docker.pkg.dev/scenic-dynamo-447811-m9/weather-health/weather-health:v1` |
| Model version | `v1` |
| Database | Cloud SQL Postgres `weather-health-db` |
| Secret | `weather-health-db-url` |
| Max instances | `5` |

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
make gcp-smoke       # health, ready, version, predict, feedback, stats
make gcp-status      # Cloud Run, Cloud SQL, secret, registry
make gcp-logs        # recent Cloud Run logs
make gcp-deploy      # build amd64 image + deploy
make gcp-deploy-current  # fetch champion, then build + deploy
make gcp-destroy CONFIRM=gcp-demo   # delete only the demo resources
make gcp-terraform-destroy          # Terraform-managed resources only
```

`gcp-destroy` refuses to run unless `CONFIRM=gcp-demo` is set. It does not
touch `elyra-*` services, Terraform state, or unrelated IAM. `gcp-smoke`
resolves the live URL from Cloud Run, so it keeps working if the service URL
changes. `gcp-deploy` reads `MODEL_VERSION` from `models/model_metadata.json`
when available; set `FETCH_MODEL=1` or use `gcp-deploy-current` to fetch the
registered champion first.

## Smoke test

```bash
URL=https://weather-health-587752925399.us-central1.run.app
curl -s "$URL/health"
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

## Required permissions for a clean deploy

The current `terraform@` service account can create Artifact Registry
repositories, deploy Cloud Run, manage buckets, and create secrets. The Cloud
Run runtime service account needs `roles/cloudsql.client` and
`roles/secretmanager.secretAccessor`.

`infra-gcp/` now models Cloud SQL, the Secret Manager secret, the runtime
service account, and the Cloud Run connector. The live resources were created
with `gcloud` first; see `infra-gcp/README.md` for import commands before
treating Terraform as the source of truth.
