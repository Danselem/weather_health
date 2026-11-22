#!/usr/bin/env bash
# Recreate the GCP demo data plane and IAM after a teardown.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gcp_env.sh
source scripts/gcp_env.sh

require_command gcloud
require_command openssl

RUNTIME_EMAIL="${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Enabling required APIs"
gcloud services enable \
  run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com sqladmin.googleapis.com iam.googleapis.com \
  --project "$PROJECT_ID" --quiet

if gcloud sql instances describe "$SQL_INSTANCE" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Cloud SQL already exists: $SQL_INSTANCE"
else
  echo "Creating Cloud SQL instance: $SQL_INSTANCE (this takes a few minutes)"
  DB_PASSWORD=$(openssl rand -hex 24)
  gcloud sql instances create "$SQL_INSTANCE" \
    --database-version=POSTGRES_16 \
    --tier=db-f1-micro \
    --edition=ENTERPRISE \
    --region="$REGION" \
    --storage-size=10GB \
    --storage-type=SSD \
    --availability-type=ZONAL \
    --no-backup \
    --root-password="$DB_PASSWORD" \
    --project "$PROJECT_ID" --quiet

  gcloud sql databases create weatherhealth \
    --instance "$SQL_INSTANCE" --project "$PROJECT_ID" --quiet || true

  CONNECTION_NAME=$(gcloud sql instances describe "$SQL_INSTANCE" \
    --project "$PROJECT_ID" --format='value(connectionName)')
  DB_URL="postgresql://postgres:${DB_PASSWORD}@/weatherhealth?host=/cloudsql/${CONNECTION_NAME}"

  if gcloud secrets describe "$SECRET_ID" --project "$PROJECT_ID" >/dev/null 2>&1; then
    printf '%s' "$DB_URL" | gcloud secrets versions add "$SECRET_ID" \
      --data-file=- --project "$PROJECT_ID" --quiet
  else
    printf '%s' "$DB_URL" | gcloud secrets create "$SECRET_ID" \
      --data-file=- --replication-policy=automatic --project "$PROJECT_ID" --quiet
  fi
fi

if ! gcloud artifacts repositories describe "$REPOSITORY" \
  --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Creating Artifact Registry repository: $REPOSITORY"
  gcloud artifacts repositories create "$REPOSITORY" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Weather-health serving images" \
    --project "$PROJECT_ID" --quiet
fi

if ! gcloud iam service-accounts describe "$RUNTIME_EMAIL" \
  --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Creating runtime service account: $RUNTIME_EMAIL"
  gcloud iam service-accounts create "$RUNTIME_SA" \
    --display-name="Cloud Run runtime for weather-health" \
    --project "$PROJECT_ID" --quiet
fi

echo "Granting runtime IAM"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_EMAIL}" \
  --role="roles/cloudsql.client" --quiet >/dev/null
gcloud secrets add-iam-policy-binding "$SECRET_ID" \
  --member="serviceAccount:${RUNTIME_EMAIL}" \
  --role="roles/secretmanager.secretAccessor" \
  --project "$PROJECT_ID" --quiet >/dev/null

echo "Bootstrap complete."
gcloud sql instances describe "$SQL_INSTANCE" --project "$PROJECT_ID" \
  --format='value(name,state,connectionName)'
