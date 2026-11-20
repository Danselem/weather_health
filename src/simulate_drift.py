"""Generate a drifted current window and run the monitoring flow."""

import numpy as np
import pandas as pd

from src.config import load_config
from src.evidently_metrics import monitoring_flow


def main() -> None:
    cfg = load_config("config")
    reference = pd.read_parquet(cfg.artifacts.reference_path)
    current = reference.copy()
    feature_columns = [
        c
        for c in reference.columns
        if c not in ("prediction", "prediction_confidence", "true_label")
    ]

    rng = np.random.default_rng(7)
    for column in feature_columns:
        current[column] = current[column].astype(float) * 1.6 + rng.normal(
            0, 2, len(current)
        )
    current["prediction"] = current["prediction"].sample(
        frac=1, random_state=7
    ).values

    out = "data/monitoring/simulated_current.parquet"
    current.to_parquet(out, index=False)
    print(f"Simulated drifted window written to {out}")
    metrics = monitoring_flow(current_path=out, fail_on_drift=False)
    print(f"Simulated drift status: {metrics['status']}")


if __name__ == "__main__":
    main()
