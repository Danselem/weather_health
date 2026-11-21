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

The live service uses ephemeral SQLite (`PREDICTION_DB_URL` unset) and
`max-instances=1`. Prediction logs reset on cold start. For persistent
monitoring, set `PREDICTION_DB_URL` to a managed Postgres URL and remove the
single-instance cap.

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
  --max-instances 1 \
  --min-instances 0 \
  --set-env-vars MODEL_VERSION="$TAG",ENVIRONMENT=prod
```

## Smoke test

```bash
URL=https://weather-health-587752925399.us-central1.run.app
curl -s "$URL/health"
curl -s "$URL/ready"
curl -s "$URL/version"
curl -s -X POST "$URL/predict" \
  -H 'Content-Type: application/json' --data @sample_input.json
```

## Terraform path

`infra-gcp/` is an alternative reference stack. It still uses local state and
does not yet mirror the live direct `gcloud` deployment. Before using it:

- replace placeholder `project_id` values in `infra-gcp/environments/*`;
- add a GCS backend;
- remove public IAM from the GCS artifacts bucket;
- add Secret Manager-backed `PREDICTION_DB_URL`;
- decide whether Terraform or `gcloud run deploy` owns image updates.

## Required permissions for a clean deploy

The current `terraform@` service account can create Artifact Registry
repositories, deploy Cloud Run, and manage buckets. A persistent Postgres
secret requires `roles/secretmanager.admin` (or a pre-created secret plus
`roles/secretmanager.secretAccessor`).
