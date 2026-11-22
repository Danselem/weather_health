#!/usr/bin/env bash
# Build and deploy the scheduled Evidently monitoring job.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gcp_env.sh
source scripts/gcp_env.sh

require_command gcloud
RUNTIME_EMAIL="${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
JOB_NAME="${GCP_MONITORING_JOB:-weather-health-monitoring}"
SCHEDULER_NAME="${GCP_MONITORING_SCHEDULER:-weather-health-monitoring-daily}"
TAG="${TAG:-$(date -u +%Y%m%d%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/weather-health:monitoring-${TAG}"

gcloud services enable cloudscheduler.googleapis.com --project "$PROJECT_ID" --quiet

echo "Building monitoring image with Cloud Build"
gcloud builds submit \
  --config monitoring/cloudbuild.yaml \
  --substitutions "_IMAGE=${IMAGE}" \
  --project "$PROJECT_ID" \
  . >/tmp/gcp_monitoring_build.log

echo "Deploying Cloud Run Job $JOB_NAME"
gcloud run jobs deploy "$JOB_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --service-account "$RUNTIME_EMAIL" \
  --set-cloudsql-instances "${PROJECT_ID}:${REGION}:${SQL_INSTANCE}" \
  --set-secrets "PREDICTION_DB_URL=${SECRET_ID}:latest" \
  --set-env-vars ENVIRONMENT=prod \
  --max-retries 1 \
  --task-timeout 600s \
  --quiet

echo "Granting scheduler invoker role"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_EMAIL}" \
  --role="roles/run.invoker" --quiet >/dev/null

echo "Creating/updating daily scheduler job"
if gcloud scheduler jobs describe "$SCHEDULER_NAME" \
  --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud scheduler jobs update http "$SCHEDULER_NAME" \
    --location "$REGION" \
    --schedule "0 6 * * *" \
    --time-zone UTC \
    --uri "https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB_NAME}:run" \
    --http-method POST \
    --update-headers "Content-Type=application/json" \
    --message-body "{}" \
    --oauth-service-account-email "$RUNTIME_EMAIL" \
    --project "$PROJECT_ID" --quiet
else
  gcloud scheduler jobs create http "$SCHEDULER_NAME" \
    --location "$REGION" \
    --schedule "0 6 * * *" \
    --time-zone UTC \
    --uri "https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB_NAME}:run" \
    --http-method POST \
    --headers "Content-Type=application/json" \
    --message-body "{}" \
    --oauth-service-account-email "$RUNTIME_EMAIL" \
    --project "$PROJECT_ID" --quiet
fi

echo "Monitoring deployed."
echo "Run now: make gcp-monitoring-run"
