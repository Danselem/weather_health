"""Calibration metrics and reliability diagram for the champion model."""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> tuple[float, pd.DataFrame]:
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, bins) - 1, 0, n_bins - 1)

    rows = []
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        avg_conf = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(avg_conf - accuracy)
        rows.append(
            {
                "bin": bin_id,
                "confidence": avg_conf,
                "accuracy": accuracy,
                "count": int(mask.sum()),
            }
        )
    return float(ece), pd.DataFrame(rows)


def main() -> None:
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    with open(params["artifacts"]["model_path"], "rb") as file:
        model = pickle.load(file)
    x_test = pd.read_csv(params["data"]["x_test_path"])
    y_test = pd.read_csv(params["data"]["y_test_path"]).values.ravel()

    probs = model.predict_proba(x_test)
    preds = probs.argmax(axis=1)
    brier = multiclass_brier(y_test, probs)
    ece, bins = expected_calibration_error(y_test, probs)

    metrics = pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(y_test, preds),
                "brier_score": brier,
                "ece": ece,
                "n_samples": len(y_test),
            }
        ]
    )
    out = Path("reports/calibration_metrics.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, index=False)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.plot(bins["confidence"], bins["accuracy"], "o-", label="Model")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title(f"Reliability diagram (ECE={ece:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/reliability_diagram.png")
    plt.close()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
