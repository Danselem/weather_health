"""Compute model performance from matured /feedback labels."""

import os
from datetime import datetime, timezone
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from src import prediction_store


def compute_performance(limit: int = 5000) -> dict[str, Any] | None:
    labeled = prediction_store.fetch_labeled_predictions(limit=limit)
    if labeled.empty:
        return None
    min_labels = int(os.getenv("MONITORING_MIN_LABELS", "20"))
    if len(labeled) < min_labels:
        print(f"insufficient_labels: {len(labeled)} labels (minimum {min_labels})")
        return None

    y_true = labeled["true_label"].tolist()
    y_pred = labeled["prediction"].tolist()
    metrics = {
        "timestamp": datetime.now(timezone.utc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "labeled_count": int(len(labeled)),
        "model_version": str(labeled["model_version"].iloc[-1]),
    }
    prediction_store.save_performance_metrics(metrics)
    print(
        f"performance accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} labeled={metrics['labeled_count']}"
    )
    return metrics


def main() -> None:
    metrics = compute_performance()
    if metrics is None:
        print("No labeled predictions; call /feedback first.")


if __name__ == "__main__":
    main()
