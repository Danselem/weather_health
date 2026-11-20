"""FastAPI service for weather-disease prediction."""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/weather_disease_model.pkl"))
ENCODER_PATH = Path(os.getenv("ENCODER_PATH", "data/processed/label_encoder.pkl"))
FEATURE_ORDER_PATH = Path(
    os.getenv("FEATURE_ORDER_PATH", "data/processed/feature_order.json")
)
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")

app = FastAPI(
    title="Weather-Disease Prediction API",
    version="1.0.0",
    openapi_tags=[
        {"name": "Health", "description": "Health and readiness checks"},
        {"name": "Prediction", "description": "Model inference"},
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


class PredictionRequest(BaseModel):
    """Raw, unscaled feature payload."""

    model_config = ConfigDict(extra="allow")

    features: dict[str, float] | None = Field(
        default=None, description="Feature name to raw value mapping"
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


@app.post("/predict", tags=["Prediction"])
def predict(payload: PredictionRequest) -> dict[str, Any]:
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
        }

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(input_df))[0]
            order = np.argsort(probabilities)[::-1][:3]
            result["confidence"] = float(probabilities[order[0]])
            result["top_k"] = [
                {"label": encoder.inverse_transform([int(i)])[0], "probability": float(probabilities[i])}
                for i in order
            ]

        return result
    except HTTPException:
        raise
    except Exception as err:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from err


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
