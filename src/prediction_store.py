"""Prediction logging with SQLite locally and Postgres in the demo stack."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SQLITE_PATH = "data/monitoring/predictions.db"


def _db_url() -> str:
    return os.getenv("PREDICTION_DB_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")


def _is_postgres() -> bool:
    return _db_url().startswith(("postgres://", "postgresql://"))


def _sqlite_path() -> Path:
    url = _db_url()
    path = url.replace("sqlite:///", "", 1) if url.startswith("sqlite:///") else url
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def _connect() -> Any:
    if _is_postgres():
        import psycopg

        return psycopg.connect(_db_url())
    return sqlite3.connect(_sqlite_path())


def _placeholder() -> str:
    return "%s" if _is_postgres() else "?"


def init_db() -> None:
    timestamp_type = "TIMESTAMPTZ" if _is_postgres() else "TEXT"
    predictions_ddl = f"""
    CREATE TABLE IF NOT EXISTS predictions (
        id TEXT PRIMARY KEY,
        timestamp {timestamp_type} NOT NULL,
        model_version TEXT,
        features TEXT NOT NULL,
        prediction TEXT,
        confidence DOUBLE PRECISION,
        top_k TEXT,
        latency_ms DOUBLE PRECISION,
        status TEXT,
        true_label TEXT
    )
    """
    metrics_ddl = f"""
    CREATE TABLE IF NOT EXISTS monitoring_metrics (
        timestamp {timestamp_type} NOT NULL,
        prediction_drift DOUBLE PRECISION,
        num_drifted_columns INTEGER,
        share_missing_values DOUBLE PRECISION,
        failed_tests INTEGER,
        status TEXT
    )
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(predictions_ddl)
        cur.execute(metrics_ddl)
        conn.commit()


def log_prediction(record: dict[str, Any]) -> None:
    init_db()
    timestamp = record.get("timestamp") or datetime.now(timezone.utc)
    values = (
        record["request_id"],
        timestamp if _is_postgres() else str(timestamp),
        record.get("model_version", "unknown"),
        json.dumps(record.get("features", {})),
        record.get("prediction"),
        record.get("confidence"),
        json.dumps(record.get("top_k", [])),
        record.get("latency_ms"),
        record.get("status", "ok"),
        record.get("true_label"),
    )
    sql = f"""
    INSERT INTO predictions
    (id, timestamp, model_version, features, prediction, confidence,
     top_k, latency_ms, status, true_label)
    VALUES ({", ".join([_placeholder()] * 10)})
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()


def set_feedback(request_id: str, true_label: str) -> bool:
    init_db()
    ph = _placeholder()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE predictions SET true_label = {ph} WHERE id = {ph}",
            (true_label, request_id),
        )
        updated = cur.rowcount
        conn.commit()
    return bool(updated)


def fetch_predictions(limit: int = 5000) -> pd.DataFrame:
    init_db()
    sql = "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT " + str(int(limit))
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["features"] = df["features"].apply(json.loads)
    return df.sort_values("timestamp").reset_index(drop=True)


def save_monitoring_metrics(metrics: dict[str, Any]) -> None:
    init_db()
    timestamp = metrics.get("timestamp") or datetime.now(timezone.utc)
    values = (
        timestamp if _is_postgres() else str(timestamp),
        float(metrics.get("prediction_drift", 0.0)),
        int(metrics.get("num_drifted_columns", 0)),
        float(metrics.get("share_missing_values", 0.0)),
        int(metrics.get("failed_tests", 0)),
        metrics.get("status", "ok"),
    )
    sql = f"""
    INSERT INTO monitoring_metrics
    (timestamp, prediction_drift, num_drifted_columns,
     share_missing_values, failed_tests, status)
    VALUES ({", ".join([_placeholder()] * 6)})
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()


def stats() -> dict[str, Any]:
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), MAX(timestamp) FROM predictions")
        count, latest = cur.fetchone()
    return {"count": int(count), "latest_prediction": str(latest) if latest else None}
