"""
serve_local.py

Load preprocessing artifacts + trained model,
read a JSON sample, scale it with feature names preserved,
and print the decoded prediction.
"""

import json
import pickle
from pathlib import Path

import pandas as pd

# ── Artifact locations ───────────────────────────────────────────────────────
MODEL_PATH = Path("models/weather_disease_model.pkl")
SCALER_PATH = Path("data/processed/minmax_scaler.pkl")
ENCODER_PATH = Path("data/processed/label_encoder.pkl")
SAMPLE_JSON_PATH = Path("sample_input.json")  # raw, un‑scaled sample


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    # ── Load artifacts --------------------------------------------------------
    model = load_pickle(MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    label_encoder = load_pickle(ENCODER_PATH)

    # ── Load raw JSON sample --------------------------------------------------
    with SAMPLE_JSON_PATH.open(encoding="utf-8") as f:
        sample_dict = json.load(f)

    # ── Ensure column order matches training ----------------------------------
    feature_order = list(scaler.feature_names_in_)  # preserved by scikit‑learn
    sample_df = pd.DataFrame([sample_dict], columns=feature_order)

    # ── Scale with names intact (silences warning) ----------------------------
    scaled_df = pd.DataFrame(scaler.transform(sample_df), columns=feature_order)

    # ── Predict and decode ----------------------------------------------------
    pred_code = model.predict(scaled_df)[0]
    prediction = label_encoder.inverse_transform([pred_code])[0]

    print("Prediction:", prediction)


if __name__ == "__main__":
    main()
