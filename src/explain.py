"""SHAP feature attribution for the champion tree pipeline."""

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml  # type: ignore[import-untyped]


def main() -> None:
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    with open(params["artifacts"]["model_path"], "rb") as file:
        model = pickle.load(file)

    x_test = pd.read_csv(params["data"]["x_test_path"])
    sample = x_test.sample(n=min(200, len(x_test)), random_state=1024)
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["clf"]
    x_scaled = pd.DataFrame(scaler.transform(sample), columns=sample.columns)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(x_scaled)
    if isinstance(shap_values, list):
        importance = np.mean(
            [np.abs(values).mean(axis=0) for values in shap_values], axis=0
        )
    elif shap_values.ndim == 3:
        # LightGBM returns (n_samples, n_features, n_classes)
        importance = np.abs(shap_values).mean(axis=(0, 2))
    else:
        importance = np.abs(shap_values).mean(axis=0)

    ranking = pd.DataFrame(
        {"feature": sample.columns, "mean_abs_shap": importance}
    ).sort_values("mean_abs_shap", ascending=False)
    ranking.to_csv("reports/shap_feature_importance.csv", index=False)

    top = ranking.head(20).sort_values("mean_abs_shap")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#4c72b0")
    plt.xlabel("Mean |SHAP value| (all classes)")
    plt.title("Top SHAP feature attributions")
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png")
    plt.close()
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
