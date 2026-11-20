"""Load the model pipeline and score one raw JSON sample locally."""

import json
import pickle
from pathlib import Path

import pandas as pd

MODEL_PATH = Path("models/weather_disease_model.pkl")
ENCODER_PATH = Path("data/processed/label_encoder.pkl")
FEATURE_ORDER_PATH = Path("data/processed/feature_order.json")
SAMPLE_JSON_PATH = Path("sample_input.json")


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def main():
    model = load_pickle(MODEL_PATH)
    label_encoder = load_pickle(ENCODER_PATH)
    feature_order = json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
    sample = json.loads(SAMPLE_JSON_PATH.read_text(encoding="utf-8"))

    sample_df = pd.DataFrame([sample], columns=feature_order)
    pred_code = model.predict(sample_df)[0]
    prediction = label_encoder.inverse_transform([pred_code])[0]
    print("Prediction:", prediction)


if __name__ == "__main__":
    main()
