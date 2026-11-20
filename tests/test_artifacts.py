"""Offline smoke tests for the artifact contract and API."""

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from service.app import app

client = TestClient(app)


def test_health_and_ready():
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/version").json()["model_version"] == "unknown"


def test_artifact_contract_and_prediction():
    import pickle

    with open("models/weather_disease_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("data/processed/label_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    x_test = pd.read_csv("data/processed/x_test.csv")
    pred = model.predict(x_test.iloc[[0]])[0]
    label = encoder.inverse_transform([pred])[0]
    assert label in encoder.classes_

    payload = json.loads(Path("sample_input.json").read_text())
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in encoder.classes_
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["top_k"][0]["label"] == body["prediction"]


def test_data_contract():
    from src.validate_data import validate

    report = validate("data/interim/weather_disease.parquet", "prognosis")
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
