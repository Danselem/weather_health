"""Prediction logging with SQLite locally and Postgres in the demo stack."""

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
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
    performance_ddl = f"""
    CREATE TABLE IF NOT EXISTS performance_metrics (
        timestamp {timestamp_type} NOT NULL,
        accuracy DOUBLE PRECISION,
        macro_f1 DOUBLE PRECISION,
        labeled_count INTEGER,
        model_version TEXT
    )
    """
    requests_ddl = f"""
    CREATE TABLE IF NOT EXISTS request_events (
        timestamp {timestamp_type} NOT NULL,
        endpoint TEXT NOT NULL,
        status_code INTEGER,
        prediction_logged BOOLEAN
    )
    """
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(true_label)",
    ]
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(predictions_ddl)
        cur.execute(metrics_ddl)
        cur.execute(performance_ddl)
        cur.execute(requests_ddl)
        for statement in indexes:
            cur.execute(statement)
        conn.commit()


def _write_dead_letter(record: dict[str, Any], error: Exception) -> None:
    path = Path(os.getenv("DEAD_LETTER_PATH", "data/monitoring/dead_letter.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": str(error),
        "record": record,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


def log_prediction(record: dict[str, Any], retries: int = 2) -> None:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            _log_prediction_once(record)
            return
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(0.2 * (attempt + 1))
    if last_error is not None:
        _write_dead_letter(record, last_error)
        raise last_error


def _log_prediction_once(record: dict[str, Any]) -> None:
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


def fetch_predictions(
    limit: int = 5000,
    since: datetime | None = None,
    status: str | None = "ok",
) -> pd.DataFrame:
    init_db()
    conditions: list[str] = []
    params: list = []
    if status is not None:
        conditions.append("status = " + _placeholder())
        params.append(status)
    if since is not None:
        conditions.append("timestamp >= " + _placeholder())
        params.append(since if _is_postgres() else str(since))
    sql = "SELECT * FROM predictions"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY timestamp DESC LIMIT " + str(int(limit))
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
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


def fetch_labeled_predictions(limit: int = 5000) -> pd.DataFrame:
    init_db()
    sql = (
        "SELECT * FROM predictions WHERE true_label IS NOT NULL "
        "ORDER BY timestamp DESC LIMIT " + str(int(limit))
    )
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns).sort_values("timestamp").reset_index(drop=True)


def save_performance_metrics(metrics: dict[str, Any]) -> None:
    init_db()
    timestamp = metrics.get("timestamp") or datetime.now(timezone.utc)
    values = (
        timestamp if _is_postgres() else str(timestamp),
        float(metrics.get("accuracy", 0.0)),
        float(metrics.get("macro_f1", 0.0)),
        int(metrics.get("labeled_count", 0)),
        metrics.get("model_version", "unknown"),
    )
    sql = f"""
    INSERT INTO performance_metrics
    (timestamp, accuracy, macro_f1, labeled_count, model_version)
    VALUES ({", ".join([_placeholder()] * 5)})
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()


def recent_monitoring_statuses(limit: int = 5) -> list[str]:
    init_db()
    sql = "SELECT status FROM monitoring_metrics ORDER BY timestamp DESC LIMIT " + str(int(limit))
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql)
        return [row[0] for row in cur.fetchall()]


def failed_streak() -> int:
    streak = 0
    for status in recent_monitoring_statuses(limit=20):
        if status != "failed":
            break
        streak += 1
    return streak


def prune_predictions(days: int = 30) -> int:
    init_db()
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    cutoff_dt = datetime.fromtimestamp(cutoff, tz=timezone.utc)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM predictions WHERE timestamp < {_placeholder()}",
            (cutoff_dt if _is_postgres() else str(cutoff_dt),),
        )
        deleted = cur.rowcount
        conn.commit()
    return int(deleted)


def log_request_event(
    *,
    endpoint: str,
    status_code: int,
    prediction_logged: bool,
    timestamp: datetime | None = None,
) -> None:
    try:
        init_db()
        ts = timestamp or datetime.now(timezone.utc)
        values = (
            ts if _is_postgres() else str(ts),
            endpoint,
            int(status_code),
            bool(prediction_logged),
        )
        sql = f"""
        INSERT INTO request_events
        (timestamp, endpoint, status_code, prediction_logged)
        VALUES ({", ".join([_placeholder()] * 4)})
        """
        with _connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, values)
            conn.commit()
    except Exception:
        # Request logging must never break the API response.
        pass


def request_summary(window_hours: int = 24) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    frame = _fetch(
        "SELECT COUNT(*) AS requests, "
        "SUM(CASE WHEN prediction_logged THEN 1 ELSE 0 END) AS logged "
        "FROM request_events WHERE timestamp >= " + _placeholder(),
        (cutoff if _is_postgres() else str(cutoff),),
    )
    if frame.empty:
        return {"requests": 0, "logged": 0, "coverage": None}
    row = frame.iloc[0]
    requests = int(row["requests"] or 0)
    logged = int(row["logged"] or 0)
    return {
        "requests": requests,
        "logged": logged,
        "coverage": (logged / requests) if requests else None,
    }


def stats() -> dict[str, Any]:
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), MAX(timestamp) FROM predictions")
        count, latest = cur.fetchone()
    return {"count": int(count), "latest_prediction": str(latest) if latest else None}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prune = sub.add_parser("prune")
    prune.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    if args.command == "prune":
        print(f"Deleted {prune_predictions(args.days)} prediction rows")


def _fetch(sql: str, params: tuple = ()) -> pd.DataFrame:
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def recent_monitoring_metrics(limit: int = 50) -> pd.DataFrame:
    return _fetch(
        "SELECT * FROM monitoring_metrics ORDER BY timestamp DESC LIMIT " + str(int(limit))
    ).sort_values("timestamp").reset_index(drop=True)


def recent_performance_metrics(limit: int = 50) -> pd.DataFrame:
    return _fetch(
        "SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT " + str(int(limit))
    ).sort_values("timestamp").reset_index(drop=True)


def prediction_summary() -> dict[str, Any]:
    frame = _fetch(
        "SELECT COUNT(*) AS count, MAX(timestamp) AS latest, "
        "SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors, "
        "AVG(confidence) AS avg_confidence FROM predictions"
    )
    if frame.empty:
        return {"count": 0, "latest": None, "errors": 0, "avg_confidence": None}
    row = frame.iloc[0]
    return {
        "count": int(row["count"] or 0),
        "latest": str(row["latest"]) if pd.notna(row["latest"]) else None,
        "errors": int(row["errors"] or 0),
        "avg_confidence": float(row["avg_confidence"])
        if pd.notna(row["avg_confidence"])
        else None,
    }


def monitoring_summary(window_hours: int = 24, row_limit: int = 5000) -> dict[str, Any]:
    metrics = recent_monitoring_metrics()
    performance = recent_performance_metrics()
    latest = metrics.iloc[-1].to_dict() if not metrics.empty else None
    if latest:
        latest["timestamp"] = str(latest["timestamp"])
    return {
        "latest": latest,
        "history": [
            {**row, "timestamp": str(row["timestamp"])}
            for row in metrics.to_dict(orient="records")
        ],
        "performance": [
            {**row, "timestamp": str(row["timestamp"])}
            for row in performance.to_dict(orient="records")
        ],
        "predictions": prediction_summary(),
        "coverage": request_summary(window_hours=window_hours),
        "window_hours": window_hours,
        "row_limit": row_limit,
    }
