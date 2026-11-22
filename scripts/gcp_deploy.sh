#!/usr/bin/env bash
# Build (linux/amd64) and deploy the demo image to Cloud Run.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gcp_env.sh
source scripts/gcp_env.sh

require_command docker
require_command gcloud
require_command jq

if [ "${FETCH_MODEL:-0}" = "1" ]; then
  uv run python -m src.fetch_best_model lightgbm
fi

RUNTIME_EMAIL="${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud sql instances describe "$SQL_INSTANCE" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Cloud SQL is missing. Run: make gcp-bootstrap" >&2
  exit 1
fi
if ! gcloud secrets describe "$SECRET_ID" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Secret $SECRET_ID is missing. Run: make gcp-bootstrap" >&2
  exit 1
fi
if ! gcloud artifacts repositories describe "$REPOSITORY" \
  --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Artifact Registry $REPOSITORY is missing. Run: make gcp-bootstrap" >&2
  exit 1
fi
if ! gcloud iam service-accounts describe "$RUNTIME_EMAIL" \
  --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Runtime service account is missing. Run: make gcp-bootstrap" >&2
  exit 1
fi

MODEL_VERSION="${MODEL_VERSION:-$(jq -r '.version // "v1"' models/model_metadata.json 2>/dev/null || echo v1)}"
TAG="${TAG:-$(date -u +%Y%m%d%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/weather-health:${TAG}"

if docker info >/dev/null 2>&1; then
  echo "Building $IMAGE (linux/amd64) with local Docker"
  docker build --platform linux/amd64 \
    --build-arg MODEL_VERSION="$MODEL_VERSION" -t "$IMAGE" .
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
  docker push "$IMAGE"
else
  echo "Docker daemon unavailable; building with Cloud Build"
  gcloud builds submit --tag "$IMAGE" --project "$PROJECT_ID" .
fi

echo "Deploying to Cloud Run"
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 5 \
  --min-instances 0 \
  --service-account "$RUNTIME_EMAIL" \
  --add-cloudsql-instances "${PROJECT_ID}:${REGION}:${SQL_INSTANCE}" \
  --set-env-vars "MODEL_VERSION=${MODEL_VERSION},ENVIRONMENT=prod" \
  --set-secrets "PREDICTION_DB_URL=${SECRET_ID}:latest" \
  --project "$PROJECT_ID" \
  --quiet

URL=$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT_ID" \
  --format='value(status.url)')
echo "Deployed: $URL"
