#!/usr/bin/env bash
# Roll back an ECS service to a known-good task definition revision.
# Usage: ./scripts/rollback.sh <cluster> <service> <task-family> <revision>
set -euo pipefail

CLUSTER="${1:?cluster required}"
SERVICE="${2:?service required}"
TASK_FAMILY="${3:?task family required}"
REVISION="${4:?revision required}"

aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --task-definition "${TASK_FAMILY}:${REVISION}" \
  --force-new-deployment

aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE"
echo "Rolled back ${SERVICE} to ${TASK_FAMILY}:${REVISION}"
