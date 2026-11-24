#!/usr/bin/env bash
# Seed the live API with reference-like rows so monitoring has a real window.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gcp_env.sh
source scripts/gcp_env.sh

require_command gcloud
SERVICE_URL=$(resolve_service_url)
COUNT="${SEED_COUNT:-200}"
WORKERS="${SEED_WORKERS:-10}"

echo "Seeding $COUNT predictions into $SERVICE_URL"
uv run python -m src.seed_traffic \
  --url "$SERVICE_URL" \
  --count "$COUNT" \
  --workers "$WORKERS"
