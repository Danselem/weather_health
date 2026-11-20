"""Stratified cross-validation for the champion pipeline."""

import pickle

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate


def main() -> None:
    import yaml  # type: ignore[import-untyped]

    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    with open(params["artifacts"]["model_path"], "rb") as file:
        model = pickle.load(file)

    x_train = pd.read_csv(params["data"]["x_train_path"])
    y_train = pd.read_csv(params["data"]["y_train_path"]).values.ravel()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1024)
    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        n_jobs=-1,
    )

    rows = []
    for metric in ("test_accuracy", "test_f1_macro"):
        values = scores[metric]
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        ci = 1.96 * std / (len(values) ** 0.5)
        rows.append(
            {
                "metric": metric.replace("test_", ""),
                "mean": mean,
                "std": std,
                "ci95_low": mean - ci,
                "ci95_high": mean + ci,
                "folds": len(values),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv("reports/cv_metrics.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
