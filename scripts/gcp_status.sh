#!/usr/bin/env bash
# Show the status of the live GCP demo resources.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gcp_env.sh
source scripts/gcp_env.sh

require_command gcloud

echo "--- Cloud Run ---"
gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT_ID" \
  --format='value(status.url,status.latestReadyRevisionName,status.conditions[0].status)' 2>&1 || true

echo "--- Cloud SQL ---"
gcloud sql instances describe "$SQL_INSTANCE" \
  --project "$PROJECT_ID" \
  --format='value(state,connectionName,settings[0].tier)' 2>&1 || true

echo "--- Secret ---"
gcloud secrets describe "$SECRET_ID" --project "$PROJECT_ID" \
  --format='value(name,createTime)' 2>&1 || true

echo "--- Artifact Registry ---"
gcloud artifacts repositories describe "$REPOSITORY" \
  --location "$REGION" --project "$PROJECT_ID" \
  --format='value(name,format)' 2>&1 || true

echo "--- API ---"
SERVICE_URL=$(resolve_service_url)
curl -fsS "$SERVICE_URL/health" 2>&1 || true
echo
