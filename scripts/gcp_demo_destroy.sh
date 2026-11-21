#!/usr/bin/env bash
# Delete only the live weather-health demo resources.
# Usage: CONFIRM=gcp-demo ./scripts/gcp_demo_destroy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gcp_env.sh
source scripts/gcp_env.sh

if [ "${CONFIRM:-}" != "gcp-demo" ]; then
  cat <<EOF
Refusing to destroy live resources.

This will delete:
  Cloud Run:          $SERVICE ($REGION)
  Cloud SQL:          $SQL_INSTANCE
  Secret:             $SECRET_ID
  Artifact Registry:  $REPOSITORY

It will NOT delete elyra-* services, Terraform state, or unrelated IAM.
Re-run with: CONFIRM=gcp-demo make gcp-destroy
EOF
  exit 1
fi

require_command gcloud

echo "Deleting Cloud Run service: $SERVICE"
gcloud run services delete "$SERVICE" --region "$REGION" --project "$PROJECT_ID" --quiet || true

echo "Deleting Cloud SQL instance: $SQL_INSTANCE"
gcloud sql instances delete "$SQL_INSTANCE" --project "$PROJECT_ID" --quiet || true

echo "Deleting secret: $SECRET_ID"
gcloud secrets delete "$SECRET_ID" --project "$PROJECT_ID" --quiet || true

echo "Deleting Artifact Registry repository: $REPOSITORY"
gcloud artifacts repositories delete "$REPOSITORY" \
  --location "$REGION" --project "$PROJECT_ID" --quiet || true

if gcloud iam service-accounts describe "$RUNTIME_SA@$PROJECT_ID.iam.gserviceaccount.com" \
  --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Deleting runtime service account: $RUNTIME_SA"
  gcloud iam service-accounts delete "$RUNTIME_SA@$PROJECT_ID.iam.gserviceaccount.com" \
    --project "$PROJECT_ID" --quiet || true
else
  echo "Runtime service account not present: $RUNTIME_SA"
fi

echo "Demo resources removed."
echo "Note: project-level IAM bindings added for the demo are not removed automatically."
