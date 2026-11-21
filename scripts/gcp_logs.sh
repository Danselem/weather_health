#!/usr/bin/env bash
# Tail recent Cloud Run logs for the demo service.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gcp_env.sh
source scripts/gcp_env.sh

require_command gcloud
LIMIT="${LOG_LIMIT:-100}"

gcloud run services logs read "$SERVICE" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --limit "$LIMIT"
