#!/usr/bin/env bash
# Apply cost/safety guardrails to the live GCP demo.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gcp_env.sh
source scripts/gcp_env.sh

require_command gcloud
BILLING_ACCOUNT_ID="${BILLING_ACCOUNT_ID:-01DDB3-DA5B0F-916817}"
BUDGET_NAME="${BUDGET_NAME:-Weather Health GCP demo}"
BUDGET_AMOUNT="${BUDGET_AMOUNT:-25USD}"

echo "Enabling Cloud SQL backups + deletion protection"
gcloud sql instances patch "$SQL_INSTANCE" \
  --backup-start-time=03:00 \
  --retained-backups-count=7 \
  --deletion-protection \
  --project "$PROJECT_ID" --quiet

gcloud services enable billingbudgets.googleapis.com --project "$PROJECT_ID" --quiet
if gcloud billing budgets list --billing-account "$BILLING_ACCOUNT_ID" \
  --format='value(displayName)' 2>/dev/null | grep -Fxq "$BUDGET_NAME"; then
  echo "Budget already exists: $BUDGET_NAME"
else
  echo "Creating budget: $BUDGET_NAME ($BUDGET_AMOUNT)"
  gcloud billing budgets create \
    --billing-account "$BILLING_ACCOUNT_ID" \
    --display-name "$BUDGET_NAME" \
    --budget-amount "$BUDGET_AMOUNT" \
    --threshold-rule=percent=0.5 \
    --threshold-rule=percent=0.8 \
    --threshold-rule=percent=1.0 \
    --quiet
fi

echo "Guardrails applied."
