"""Load-test profile for the weather-health API.

Run against a local server with rate limiting disabled:

    RATE_LIMIT_PER_MINUTE=0 uv run uvicorn service.app:app --port 18084
    LOCUST_HOST=http://127.0.0.1:18084 make loadtest
"""

import json
import os
from pathlib import Path

from locust import HttpUser, between, task

PAYLOAD = json.loads(
    (Path(__file__).parent.parent / "sample_input.json").read_text(encoding="utf-8")
)
API_KEY = os.getenv("LOADTEST_API_KEY", "")


class WeatherHealthUser(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self) -> None:
        self.last_request_id: str | None = None
        self.headers = {"Content-Type": "application/json"}
        if API_KEY:
            self.headers["X-API-Key"] = API_KEY

    @task(10)
    def predict(self) -> None:
        with self.client.post(
            "/predict", json=PAYLOAD, headers=self.headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                self.last_request_id = response.json().get("request_id")
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def feedback(self) -> None:
        if not self.last_request_id:
            return
        with self.client.post(
            "/feedback",
            json={"request_id": self.last_request_id, "true_label": "Heat Stroke"},
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health(self) -> None:
        self.client.get("/health")

    @task(1)
    def stats(self) -> None:
        self.client.get("/stats")
