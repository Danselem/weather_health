"""Error and fairness slices by age band and gender."""

import pickle
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def _metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main() -> None:
    import yaml  # type: ignore[import-untyped]

    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    with open(params["artifacts"]["model_path"], "rb") as file:
        model = pickle.load(file)
    with open(params["artifacts"]["label_encoder_path"], "rb") as file:
        encoder = pickle.load(file)

    x_test = pd.read_csv(params["data"]["x_test_path"])
    y_test = pd.read_csv(params["data"]["y_test_path"]).values.ravel()
    y_true = encoder.inverse_transform(y_test)
    y_pred = encoder.inverse_transform(model.predict(x_test))

    frame = x_test.copy()
    frame["y_true"] = y_true
    frame["y_pred"] = y_pred
    frame["age_band"] = pd.cut(
        frame["Age"],
        bins=[-1, 29, 44, 59, 200],
        labels=["<30", "30-44", "45-59", "60+"],
    )
    frame["gender"] = frame["Gender"].map({0: "0", 1: "1"})

    rows = []
    for dimension in ("age_band", "gender"):
        for value, group in frame.groupby(dimension, observed=True):
            result = _metrics(group["y_true"], group["y_pred"])
            rows.append(
                {
                    "slice_type": dimension,
                    "slice_value": str(value),
                    **result,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv("reports/slice_metrics.csv", index=False)

    for dimension in ("age_band", "gender"):
        subset = out[out["slice_type"] == dimension]
        print(
            f"{dimension}: accuracy gap={subset['accuracy'].max() - subset['accuracy'].min():.4f} "
            f"macro_f1 gap={subset['macro_f1'].max() - subset['macro_f1'].min():.4f}"
        )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
