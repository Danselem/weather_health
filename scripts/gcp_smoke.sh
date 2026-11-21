#!/usr/bin/env bash
# Smoke-test the live Cloud Run API.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gcp_env.sh
source scripts/gcp_env.sh

require_command curl
require_command python3

echo "URL: $SERVICE_URL"

echo -n "health:  "
curl -fsS "$SERVICE_URL/health"
echo

echo -n "ready:   "
curl -fsS "$SERVICE_URL/ready"
echo

echo -n "version: "
curl -fsS "$SERVICE_URL/version"
echo

PREDICTION=$(curl -fsS -X POST "$SERVICE_URL/predict" \
  -H 'Content-Type: application/json' --data @sample_input.json)
echo "predict: $PREDICTION"

REQUEST_ID=$(printf '%s' "$PREDICTION" | python3 -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
LABEL=$(printf '%s' "$PREDICTION" | python3 -c 'import sys,json; print(json.load(sys.stdin)["prediction"])')

echo -n "feedback: "
curl -fsS -X POST "$SERVICE_URL/feedback" \
  -H 'Content-Type: application/json' \
  -d "{\"request_id\":\"$REQUEST_ID\",\"true_label\":\"$LABEL\"}"
echo

echo -n "stats:    "
curl -fsS "$SERVICE_URL/stats"
echo

echo "docs:     $SERVICE_URL/docs"
