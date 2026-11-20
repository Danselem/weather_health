"""Optional webhook alerts for failed drift/performance checks."""

import json
import logging
import os
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def format_alert(metrics: dict[str, Any]) -> str:
    return (
        "Weather-health monitoring alert\n"
        f"status={metrics.get('status')} "
        f"failed_tests={metrics.get('failed_tests')} "
        f"drifted_columns={metrics.get('num_drifted_columns')} "
        f"prediction_drift={metrics.get('prediction_drift')}"
    )


def send_alert(metrics: dict[str, Any]) -> bool:
    url = os.getenv("ALERT_WEBHOOK_URL")
    if not url:
        logger.warning("ALERT_WEBHOOK_URL not set; alert not delivered")
        return False

    payload = {
        "text": format_alert(metrics),
        "metrics": {k: v for k, v in metrics.items() if k != "failed_test_details"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10):
            return True
    except Exception:
        logger.exception("Failed to deliver alert")
        return False
