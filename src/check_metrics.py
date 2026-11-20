"""Fail CI when the evaluated model or data contract regresses."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--min-f1", type=float, default=0.90)
    args = parser.parse_args()

    metrics = pd.read_csv("reports/evaluation_metrics.csv").iloc[0]
    validation = json.loads(Path("reports/data_validation.json").read_text())
    ablation = pd.read_csv("reports/ablation_metrics.csv").set_index("feature_set")

    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['f1_score']:.4f}")
    print(
        "ablation: weather_only="
        f"{ablation.loc['weather_only', 'macro_f1']:.4f}, "
        f"symptoms_only={ablation.loc['symptoms_only', 'macro_f1']:.4f}"
    )

    failures = []
    if not validation.get("passed", False):
        failures.append("data validation failed")
    if metrics["accuracy"] < args.min_accuracy:
        failures.append(f"accuracy {metrics['accuracy']:.4f} < {args.min_accuracy}")
    if metrics["f1_score"] < args.min_f1:
        failures.append(f"macro F1 {metrics['f1_score']:.4f} < {args.min_f1}")
    if ablation.loc["weather_only", "macro_f1"] > 0.60:
        failures.append("weather-only baseline unexpectedly high; re-check leakage")

    if failures:
        raise SystemExit("Quality gate failed: " + "; ".join(failures))
    print("Quality gate passed.")


if __name__ == "__main__":
    main()
