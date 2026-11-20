"""Compare weather-only, symptom-only, and full feature sets."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.config import load_config

WEATHER_COLUMNS = ["Temperature (C)", "Humidity", "Wind Speed (km/h)"]


def evaluate_feature_set(
    name: str,
    columns: list[str],
    x_train: pd.DataFrame,
    y_train: Any,
    x_test: pd.DataFrame,
    y_test: Any,
) -> dict[str, Any]:
    model = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", RandomForestClassifier(n_estimators=120, random_state=1024, n_jobs=-1)),
        ]
    )
    model.fit(x_train[columns], y_train)
    pred = model.predict(x_test[columns])
    return {
        "feature_set": name,
        "n_features": len(columns),
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
    }


def main() -> None:
    cfg = load_config("config")
    x_train = pd.read_csv(cfg.data.x_train_path)
    y_train = pd.read_csv(cfg.data.y_train_path).values.ravel()
    x_test = pd.read_csv(cfg.data.x_test_path)
    y_test = pd.read_csv(cfg.data.y_test_path).values.ravel()

    all_features = list(x_train.columns)
    symptoms = [c for c in all_features if c not in WEATHER_COLUMNS]

    rows = [
        evaluate_feature_set("weather_only", WEATHER_COLUMNS, x_train, y_train, x_test, y_test),
        evaluate_feature_set("symptoms_only", symptoms, x_train, y_train, x_test, y_test),
        evaluate_feature_set("full", all_features, x_train, y_train, x_test, y_test),
    ]
    out = Path(cfg.reports.ablation_metrics_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
