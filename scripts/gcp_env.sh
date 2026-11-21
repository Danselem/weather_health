#!/usr/bin/env bash
# Shared GCP demo configuration. Source this file, do not execute it.
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-scenic-dynamo-447811-m9}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${GCP_SERVICE:-weather-health}"
SQL_INSTANCE="${GCP_SQL_INSTANCE:-weather-health-db}"
SECRET_ID="${GCP_SECRET_ID:-weather-health-db-url}"
REPOSITORY="${GCP_REPOSITORY:-weather-health}"
RUNTIME_SA="${GCP_RUNTIME_SA:-weather-health-prod-runtime}"
SERVICE_URL="${GCP_SERVICE_URL:-}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 1
  }
}

resolve_service_url() {
  if [ -n "${SERVICE_URL:-}" ]; then
    printf '%s' "$SERVICE_URL"
    return
  fi
  require_command gcloud
  gcloud run services describe "$SERVICE" \
    --region "$REGION" --project "$PROJECT_ID" \
    --format='value(status.url)'
}
