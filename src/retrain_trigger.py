"""Trigger a retrain workflow after repeated monitoring failures."""

import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def trigger_retrain(reason: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Notify a retrain workflow; never promote a model automatically."""
    payload = {
        "event_type": "retrain",
        "reason": reason,
        "metrics": {
            k: v for k, v in metrics.items() if k != "failed_test_details"
        },
    }

    webhook = os.getenv("RETRAIN_WEBHOOK_URL")
    if webhook:
        request = urllib.request.Request(
            webhook,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10):
                logger.info("Retrain webhook delivered")
                return {"triggered": True, "via": "webhook"}
        except Exception:
            logger.exception("Retrain webhook failed")

    repo = os.getenv("RETRAIN_GITHUB_REPO")
    token = os.getenv("RETRAIN_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")
    if repo and token:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/dispatches",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10):
                logger.info("GitHub repository_dispatch retrain delivered")
                return {"triggered": True, "via": "github_dispatch"}
        except Exception:
            logger.exception("GitHub retrain dispatch failed")

    path = Path("reports/monitoring/retrain_request.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.warning("No retrain target configured; wrote %s", path)
    return {"triggered": False, "via": "file", "path": str(path)}
