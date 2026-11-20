"""Generate portfolio figures from analysis CSVs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ablation = pd.read_csv("reports/ablation_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(ablation))
    width = 0.38
    ax.bar([i - width / 2 for i in x], ablation["accuracy"], width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], ablation["macro_f1"], width, label="Macro F1")
    ax.set_xticks(list(x), ablation["feature_set"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Feature ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig("reports/ablation_metrics.png")
    plt.close(fig)

    slices = pd.read_csv("reports/slice_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, dimension in zip(axes, ("age_band", "gender")):
        subset = slices[slices["slice_type"] == dimension]
        ax.bar(subset["slice_value"], subset["accuracy"], label="Accuracy", alpha=0.8)
        ax.bar(
            subset["slice_value"],
            subset["macro_f1"],
            alpha=0.55,
            label="Macro F1",
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(dimension.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Score")
    axes[1].legend()
    fig.suptitle("Performance slices")
    fig.tight_layout()
    fig.savefig("reports/slice_metrics.png")
    plt.close(fig)
    print("Wrote reports/ablation_metrics.png and reports/slice_metrics.png")


if __name__ == "__main__":
    main()
