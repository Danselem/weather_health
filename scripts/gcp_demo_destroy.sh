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

It will destroy Terraform-managed resources when Terraform state exists.
elyra-* services and unrelated IAM are left alone.
Re-run with: CONFIRM=gcp-demo make gcp-destroy
EOF
  exit 1
fi

require_command gcloud

STATE_COUNT=0
if command -v terraform >/dev/null 2>&1 && [ -f infra-gcp/backend.hcl ]; then
  STATE_COUNT=$(cd infra-gcp && terraform state list 2>/dev/null | wc -l | tr -d ' ')
fi

if [ "$STATE_COUNT" -gt 0 ]; then
  echo "Terraform state found; destroying managed demo resources."
  gcloud sql instances patch "$SQL_INSTANCE" --no-deletion-protection \
    --project "$PROJECT_ID" --quiet || true
  cd infra-gcp
  terraform destroy \
    -var-file=environments/prod/terraform.tfvars \
    -var="project_id=$PROJECT_ID" -var="region=$REGION" \
    -auto-approve
  exit 0
fi

echo "No Terraform state found; deleting live resources with gcloud."
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

echo "Removing demo IAM bindings"
gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client" --quiet >/dev/null 2>&1 || true
gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:587752925399-compute@developer.gserviceaccount.com" \
  --role="roles/cloudsql.client" --quiet >/dev/null 2>&1 || true

echo "Demo resources removed."
