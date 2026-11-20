"""Prediction store and drift-gate tests."""

import numpy as np
import pandas as pd

from src import prediction_store
from src.evidently_metrics import run_evidently


def test_prediction_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_DB_URL", f"sqlite:///{tmp_path}/predictions.db")
    prediction_store.init_db()
    prediction_store.log_prediction(
        {
            "request_id": "req-1",
            "model_version": "test",
            "features": {"Age": 30},
            "prediction": "Flu",
            "confidence": 0.9,
            "top_k": [{"label": "Flu", "probability": 0.9}],
            "latency_ms": 12.0,
        }
    )
    df = prediction_store.fetch_predictions()
    assert len(df) == 1
    assert df.iloc[0]["prediction"] == "Flu"
    assert prediction_store.set_feedback("req-1", "Flu") is True
    assert prediction_store.stats()["count"] == 1


def test_drift_gate_flags_shifted_data(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_DB_URL", f"sqlite:///{tmp_path}/predictions.db")
    reference = pd.read_parquet("data/processed/reference.parquet")
    current = reference.copy()
    feature_columns = [
        c
        for c in reference.columns
        if c not in ("prediction", "prediction_confidence", "true_label")
    ]
    rng = np.random.default_rng(11)
    for column in feature_columns:
        current[column] = current[column].astype(float) * 1.8 + rng.normal(
            0, 3, len(current)
        )
    current["prediction"] = current["prediction"].iloc[0]

    metrics = run_evidently.fn(reference, current, str(tmp_path / "reports"))
    assert metrics["failed_tests"] > 0
    assert metrics["status"] == "failed"
    assert any((tmp_path / "reports").glob("drift_*.html"))
