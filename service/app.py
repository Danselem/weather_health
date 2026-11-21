"""FastAPI service for weather-disease prediction."""

import json
import logging
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

try:
    from src import prediction_store
except ImportError:  # container image copies this module to /code
    import prediction_store  # type: ignore[no-redef]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/weather_disease_model.pkl"))
ENCODER_PATH = Path(os.getenv("ENCODER_PATH", "data/processed/label_encoder.pkl"))
FEATURE_ORDER_PATH = Path(
    os.getenv("FEATURE_ORDER_PATH", "data/processed/feature_order.json")
)
SAMPLE_INPUT_PATH = Path(os.getenv("SAMPLE_INPUT_PATH", "sample_input.json"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")


def load_sample_payload() -> dict[str, Any]:
    """Load the demo payload, falling back to a feature-order-based zero row."""
    try:
        with open(SAMPLE_INPUT_PATH, encoding="utf-8") as file:
            return cast(dict[str, Any], json.load(file))
    except Exception:
        try:
            with open(FEATURE_ORDER_PATH, encoding="utf-8") as file:
                return {feature: 0.0 for feature in json.load(file)}
        except Exception:
            return {"Age": 30.0, "Gender": 1.0, "Temperature (C)": 37.0}


SAMPLE_PAYLOAD = load_sample_payload()

app = FastAPI(
    title="Weather-Disease Prediction API",
    version="1.1.0",
    description=(
        "Interactive demo: [/demo](/demo). Swagger examples are prefilled with "
        "a valid raw feature row."
    ),
    openapi_tags=[
        {"name": "Health", "description": "Health, readiness, and demo UI"},
        {"name": "Prediction", "description": "Model inference and feedback"},
    ],
)

_artifacts: tuple[Any, Any, list[str]] | None = None


def load_artifacts() -> tuple[Any, Any, list[str]]:
    """Load the model pipeline and encoder once, on first use."""
    global _artifacts
    if _artifacts is None:
        for path in (MODEL_PATH, ENCODER_PATH, FEATURE_ORDER_PATH):
            if not path.exists():
                raise FileNotFoundError(f"Artifact not found: {path}")
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        with open(ENCODER_PATH, "rb") as file:
            encoder = pickle.load(file)
        with open(FEATURE_ORDER_PATH, encoding="utf-8") as file:
            feature_order = json.load(file)
        _artifacts = (model, encoder, feature_order)
        logger.info("Artifacts loaded from %s", MODEL_PATH)
    return _artifacts


class FeedbackRequest(BaseModel):
    request_id: str
    true_label: str


class PredictionRequest(BaseModel):
    """Raw, unscaled feature payload.

    Accepts either a flat feature mapping or {"features": {...}}.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"examples": [SAMPLE_PAYLOAD]},  # type: ignore[list-item]
    )

    features: dict[str, float] | None = Field(
        default=None,
        description="Feature name to raw value mapping; a flat payload also works.",
    )


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/ready", tags=["Health"])
def ready() -> dict[str, str]:
    missing = [
        str(path)
        for path in (MODEL_PATH, ENCODER_PATH, FEATURE_ORDER_PATH)
        if not path.exists()
    ]
    if missing:
        raise HTTPException(status_code=503, detail=f"Missing artifacts: {missing}")
    try:
        load_artifacts()
    except Exception as err:
        raise HTTPException(status_code=503, detail=f"Artifacts not usable: {err}") from err
    return {"status": "ready"}


@app.get("/version", tags=["Health"])
def version() -> dict[str, str]:
    return {"model_version": MODEL_VERSION}


@app.get("/sample-input", tags=["Health"])
def sample_input() -> dict[str, Any]:
    """Return a valid example payload for manual testing."""
    return SAMPLE_PAYLOAD


@app.get("/demo", response_class=HTMLResponse, tags=["Health"])
def demo() -> str:
    """Small zero-build demo UI with a prefilled payload textarea."""
    return DEMO_HTML


@app.post("/feedback", tags=["Prediction"])
def feedback(payload: FeedbackRequest) -> dict[str, Any]:
    _, encoder, _ = load_artifacts()
    if payload.true_label not in set(encoder.classes_):
        raise HTTPException(status_code=400, detail="Unknown label")
    updated = prediction_store.set_feedback(payload.request_id, payload.true_label)
    if not updated:
        raise HTTPException(status_code=404, detail="Request ID not found")
    return {"status": "recorded", "request_id": payload.request_id}


@app.get("/stats", tags=["Health"])
def prediction_stats() -> dict[str, Any]:
    return prediction_store.stats()


@app.post("/predict", tags=["Prediction"])
def predict(
    payload: PredictionRequest = Body(
        ...,
        openapi_examples={
            "sample_raw_features": {
                "summary": "Valid raw feature row",
                "description": "Prefilled from sample_input.json",
                "value": SAMPLE_PAYLOAD,
            }
        },
    ),
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        model, encoder, feature_order = load_artifacts()
        features = payload.features if payload.features is not None else payload.model_extra
        if not features:
            raise HTTPException(status_code=400, detail="No features provided")
        logger.debug("Received payload: %s", json.dumps(features))

        missing = [col for col in feature_order if col not in features]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Missing feature(s): {', '.join(missing)}"
            )

        input_df = pd.DataFrame(
            [[features[col] for col in feature_order]], columns=feature_order
        )

        # The model artifact is a full sklearn Pipeline: scaling happens inside it.
        pred_code = model.predict(input_df)[0]
        prediction = encoder.inverse_transform([pred_code])[0]

        result: dict[str, Any] = {
            "prediction": prediction,
            "model_version": MODEL_VERSION,
            "request_id": request_id,
        }

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(input_df))[0]
            order = np.argsort(probabilities)[::-1][:3]
            result["confidence"] = float(probabilities[order[0]])
            result["top_k"] = [
                {
                    "label": encoder.inverse_transform([int(i)])[0],
                    "probability": float(probabilities[i]),
                }
                for i in order
            ]

        try:
            prediction_store.log_prediction(
                {
                    "request_id": request_id,
                    "model_version": MODEL_VERSION,
                    "features": features,
                    "prediction": prediction,
                    "confidence": result.get("confidence"),
                    "top_k": result.get("top_k", []),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "status": "ok",
                }
            )
        except Exception:
            logger.exception("Prediction logging failed; returning response anyway")

        return result
    except HTTPException:
        try:
            prediction_store.log_prediction(
                {
                    "request_id": request_id,
                    "model_version": MODEL_VERSION,
                    "features": payload.model_extra or payload.features or {},
                    "status": "error",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
        except Exception:
            logger.exception("Failed to log error prediction")
        raise
    except Exception as err:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from err


DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Weather-Health Demo</title>
  <style>
    :root { color-scheme: light dark; --accent:#2563eb; --muted:#6b7280; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
           max-width: 980px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
    h1 { margin-bottom: .2rem; }
    .muted { color: var(--muted); margin-top: 0; }
    .grid { display: grid; grid-template-columns: 1.1fr .9fr; gap: 1rem; }
    textarea { width: 100%; min-height: 380px; font-family: ui-monospace, monospace;
               font-size: 12px; padding: .8rem; border: 1px solid #cbd5e1;
               border-radius: 8px; box-sizing: border-box; }
    button { background: var(--accent); color: white; border: 0; border-radius: 6px;
             padding: .55rem .9rem; margin: .25rem .25rem .25rem 0; cursor: pointer; }
    button.secondary { background: #475569; }
    pre { background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 8px;
          min-height: 300px; overflow: auto; }
    a { color: var(--accent); }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>Weather-Health Demo</h1>
  <p class="muted">Prefilled raw features. Edit the JSON, then click Predict.
    <a href="/docs">Swagger</a> · <a href="/health">health</a> · <a href="/stats">stats</a></p>
  <div class="grid">
    <div>
      <label for="payload"><strong>Request payload</strong></label>
      <textarea id="payload" spellcheck="false"></textarea>
      <div>
        <button onclick="predict()">Predict</button>
        <button class="secondary" onclick="loadSample()">Reset sample</button>
        <button class="secondary" onclick="sendFeedback()">Send feedback</button>
        <button class="secondary" onclick="loadStats()">Stats</button>
      </div>
    </div>
    <div>
      <strong>Response</strong>
      <pre id="result">Ready.</pre>
    </div>
  </div>
  <script>
    let lastRequestId = null;
    async function loadSample() {
      const response = await fetch('/sample-input');
      const data = await response.json();
      document.getElementById('payload').value = JSON.stringify(data, null, 2);
      document.getElementById('result').textContent = 'Sample loaded.';
    }
    async function predict() {
      const out = document.getElementById('result');
      try {
        const payload = JSON.parse(document.getElementById('payload').value);
        const response = await fetch('/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        lastRequestId = data.request_id || null;
        out.textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        out.textContent = 'Error: ' + error.message;
      }
    }
    async function sendFeedback() {
      const out = document.getElementById('result');
      if (!lastRequestId) {
        out.textContent = 'Run a prediction first.';
        return;
      }
      const label = prompt('True label:', 'Heat Stroke');
      if (!label) return;
      const response = await fetch('/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({request_id: lastRequestId, true_label: label})
      });
      out.textContent = JSON.stringify(await response.json(), null, 2);
    }
    async function loadStats() {
      const response = await fetch('/stats');
      document.getElementById('result').textContent =
        JSON.stringify(await response.json(), null, 2);
    }
    loadSample();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
