"""
FastAPI service for weather‑disease prediction 
(pickle model + scaler + label encoder).
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ── Artifact paths inside the Docker image ───────────────────────────────────
MODEL_PATH = Path("weather_disease_model.pkl")
SCALER_PATH = Path("minmax_scaler.pkl")
ENCODER_PATH = Path("label_encoder.pkl")

# ── Load artifacts -----------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    minmax_scaler = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

feature_order = list(minmax_scaler.feature_names_in_)  # exact training column order

# ── FastAPI app --------------------------------------------------------------
app = FastAPI(
    title="Weather‑Disease Prediction API",
    openapi_tags=[
        {"name": "Health", "description": "API health‑check"},
        {"name": "Prediction", "description": "Model inference"},
    ],
)


# Health‑check ----------------------------------------------------------------
@app.get("/", tags=["Health"])
def api_health() -> Dict[str, str]:
    return {"status": "healthy"}


# Prediction ------------------------------------------------------------------
@app.post("/predict", tags=["Prediction"])
def predict(payload: Dict) -> Dict[str, str]:
    """
    Receive JSON with feature keys, scale it, predict, and return disease label.
    """
    try:
        logger.debug("Received payload: %s", json.dumps(payload))

        # Validate input is a dict
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Input must be a JSON object")

        # Check for missing features
        missing = [col for col in feature_order if col not in payload]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Missing feature(s): {', '.join(missing)}"
            )

        # Build DataFrame in correct column order
        input_df = pd.DataFrame(
            [[payload[col] for col in feature_order]], columns=feature_order
        )
        logger.debug("Input DataFrame:\n%s", input_df)

        # Scale features and wrap back into DataFrame to keep column names
        scaled_array = minmax_scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_array, columns=feature_order)
        logger.debug("Scaled input DataFrame:\n%s", scaled_df)

        # Predict
        pred_code = model.predict(scaled_df)[0]

        # Decode numeric label to original string
        prediction = label_encoder.inverse_transform([pred_code])[0]
        logger.debug("Prediction code: %s -> label: %s", pred_code, prediction)

        return {"prediction": prediction}

    except HTTPException as http_err:
        raise http_err

    except Exception as err:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail=str(err))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
