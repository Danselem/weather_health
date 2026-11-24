"""Prediction store and drift-gate tests."""

import numpy as np
import pandas as pd

from src import prediction_store
from src.alerts import format_alert, send_alert
from src.evidently_metrics import run_evidently, run_monitoring
from src.performance_monitor import compute_performance
from src.retrain_trigger import trigger_retrain


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


def test_alert_and_retrain_fallbacks(tmp_path, monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("RETRAIN_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("RETRAIN_GITHUB_REPO", raising=False)
    monkeypatch.delenv("RETRAIN_GITHUB_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)

    metrics = {
        "status": "failed",
        "failed_tests": 3,
        "num_drifted_columns": 12,
        "prediction_drift": 0.7,
    }
    assert "failed_tests=3" in format_alert(metrics)
    assert send_alert(metrics) is False
    result = trigger_retrain("test", metrics)
    assert result["triggered"] is False
    assert (tmp_path / "reports/monitoring/retrain_request.json").exists()


def test_performance_metrics_from_feedback(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_DB_URL", f"sqlite:///{tmp_path}/predictions.db")
    monkeypatch.setenv("MONITORING_MIN_LABELS", "2")
    prediction_store.log_prediction(
        {"request_id": "a", "features": {}, "prediction": "Flu", "true_label": "Flu"}
    )
    prediction_store.log_prediction(
        {"request_id": "b", "features": {}, "prediction": "Flu", "true_label": "Cold"}
    )
    metrics = compute_performance()
    assert metrics is not None
    assert metrics["labeled_count"] == 2
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_failed_streak(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_DB_URL", f"sqlite:///{tmp_path}/predictions.db")
    for status in ("passed", "failed", "failed"):
        prediction_store.save_monitoring_metrics({"status": status, "failed_tests": 1})
    assert prediction_store.failed_streak() == 2


def test_minimum_rows_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_DB_URL", f"sqlite:///{tmp_path}/predictions.db")
    metrics = run_monitoring(min_rows=100, window_hours=24)
    assert metrics["status"] == "insufficient_data"
    assert metrics["failed_tests"] == 0
    assert metrics["current_rows"] == 0
