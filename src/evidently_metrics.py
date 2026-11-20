"""Scheduled drift report: current production window vs a pinned reference."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import MissingValueCount, ValueDrift
from evidently.presets import DataDriftPreset
from prefect import flow, task

from src import prediction_store
from src.alerts import send_alert
from src.config import load_config
from src.performance_monitor import compute_performance
from src.retrain_trigger import trigger_retrain


def _load_current(
    reference_columns: list[str], current_path: str | None, limit: int
) -> pd.DataFrame:
    if current_path:
        path = Path(current_path)
        current = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        )
    else:
        logged = prediction_store.fetch_predictions(limit=limit)
        if logged.empty:
            raise ValueError(
                "No logged predictions. Send traffic to /predict or pass --current-path."
            )
        features = pd.json_normalize(logged["features"])
        current = features.copy()
        current["prediction"] = logged["prediction"].values
        current["prediction_confidence"] = logged["confidence"].values
        if "true_label" in logged.columns:
            current["true_label"] = logged["true_label"].values

    missing = [column for column in reference_columns if column not in current.columns]
    if missing:
        raise ValueError(f"Current window is missing columns: {missing}")
    return current[reference_columns].copy()


def _extract_metrics(result: dict[str, Any]) -> dict[str, Any]:
    prediction_drift = 0.0
    drifted_columns = 0
    share_missing = 0.0

    for metric in result.get("metrics", []):
        metric_id = str(metric.get("metric_id", ""))
        value = metric.get("value")
        if "ValueDrift" in metric_id and "prediction" in metric_id:
            try:
                prediction_drift = float(value)
            except (TypeError, ValueError):
                pass
        if "DriftedColumnsCount" in metric_id and isinstance(value, dict):
            drifted_columns = int(value.get("count", 0))
        if "MissingValueCount" in metric_id and isinstance(value, dict):
            share_missing = float(value.get("share", 0.0))

    tests = result.get("tests", [])
    failed = [t for t in tests if t.get("status") in ("FAIL", "ERROR")]
    return {
        "prediction_drift": prediction_drift,
        "num_drifted_columns": drifted_columns,
        "share_missing_values": share_missing,
        "failed_tests": len(failed),
        "status": "failed" if failed else "passed",
        "failed_test_details": failed,
    }


@task(name="run_evidently_drift_report")
def run_evidently(
    reference: pd.DataFrame, current: pd.DataFrame, reports_dir: str
) -> dict[str, Any]:
    numeric_columns = [
        c
        for c in reference.columns
        if c not in ("prediction", "true_label")
    ]
    definition = DataDefinition(
        numerical_columns=numeric_columns,
        categorical_columns=["prediction"],
    )
    reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
    current_dataset = Dataset.from_pandas(current, data_definition=definition)

    report = Report(
        [
            DataDriftPreset(),
            ValueDrift(column="prediction"),
            MissingValueCount(column="prediction"),
        ],
        include_tests=True,
    )
    snapshot = report.run(
        reference_data=reference_dataset, current_data=current_dataset
    )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_dir / f"drift_{stamp}.html"))
    snapshot.save_json(str(output_dir / f"drift_{stamp}.json"))

    result = snapshot.dict()
    metrics = _extract_metrics(result)
    metrics["timestamp"] = datetime.now(timezone.utc)
    prediction_store.save_monitoring_metrics(metrics)

    if metrics["failed_tests"]:
        alert = {
            "timestamp": stamp,
            "failed_tests": metrics["failed_test_details"],
        }
        (output_dir / f"alert_{stamp}.json").write_text(
            json.dumps(alert, indent=2), encoding="utf-8"
        )
        print(f"DRIFT ALERT: {metrics['failed_tests']} test(s) failed")

    print(
        f"prediction_drift={metrics['prediction_drift']:.4f} "
        f"drifted_columns={metrics['num_drifted_columns']} "
        f"missing={metrics['share_missing_values']:.4f} "
        f"status={metrics['status']}"
    )
    return metrics


@flow(name="monitoring")
def monitoring_flow(
    current_path: str | None = None,
    limit: int = 5000,
    fail_on_drift: bool = False,
    trigger_retrain_on_failure: bool = False,
) -> dict[str, Any]:
    cfg = load_config("config")
    reference_path = Path(cfg.artifacts.reference_path)
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference not found: {reference_path}. Run `uv run -m src.evaluate`."
        )

    reference = pd.read_parquet(reference_path)
    reference_columns = [c for c in reference.columns if c != "true_label"]
    current = _load_current(reference_columns, current_path, limit)
    metrics = run_evidently(reference, current, cfg.monitoring.reports_dir)

    if metrics["failed_tests"]:
        send_alert(metrics)
        streak = prediction_store.failed_streak()
        metrics["failed_streak"] = streak
        threshold = int(os.getenv("RETRAIN_FAILURE_THRESHOLD", "2"))
        if trigger_retrain_on_failure and streak >= threshold:
            metrics["retrain"] = trigger_retrain(
                reason=f"{streak} consecutive failed monitoring windows", metrics=metrics
            )

    performance = compute_performance()
    if performance is not None:
        metrics["performance"] = performance

    if fail_on_drift and metrics["failed_tests"]:
        raise SystemExit(f"Drift gate failed: {metrics['failed_tests']} test(s)")
    return cast(dict[str, Any], metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-path", default=None)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--trigger-retrain", action="store_true")
    args = parser.parse_args()
    monitoring_flow(
        current_path=args.current_path,
        limit=args.limit,
        fail_on_drift=args.fail_on_drift,
        trigger_retrain_on_failure=args.trigger_retrain,
    )


if __name__ == "__main__":
    main()
