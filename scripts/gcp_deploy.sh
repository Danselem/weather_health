#!/usr/bin/env bash
# Build (linux/amd64) and deploy the demo image to Cloud Run.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gcp_env.sh
source scripts/gcp_env.sh

require_command docker
require_command gcloud

MODEL_VERSION="${MODEL_VERSION:-v1}"
TAG="${TAG:-$(date -u +%Y%m%d%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/weather-health:${TAG}"

echo "Building $IMAGE (linux/amd64)"
docker build --platform linux/amd64 \
  --build-arg MODEL_VERSION="$MODEL_VERSION" -t "$IMAGE" .

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker push "$IMAGE"

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
  --add-cloudsql-instances "${PROJECT_ID}:${REGION}:${SQL_INSTANCE}" \
  --set-env-vars "MODEL_VERSION=${MODEL_VERSION},ENVIRONMENT=prod" \
  --set-secrets "PREDICTION_DB_URL=${SECRET_ID}:latest" \
  --project "$PROJECT_ID" \
  --quiet

URL=$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT_ID" \
  --format='value(status.url)')
echo "Deployed: $URL"
