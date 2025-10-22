"""
plot_utils.py

Helper for plotting confusion matrices with original class names
restored via the fitted LabelEncoder.
"""

import pickle
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import yaml
from numpy.typing import ArrayLike
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

params_file = Path("params.yaml")
params = yaml.safe_load(params_file.read_text())
label_encoder_path = params["artifacts"]["label_encoder_path"]


def load_label_encoder(path: Union[str, Path]) -> LabelEncoder:
    """Load the pickled LabelEncoder created during preprocessing."""
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    split: str,
    label_encoder_path: Union[str, Path] = label_encoder_path,
) -> None:
    """
    Plot and save a confusion matrix with human‑readable class names.

    Args:
        y_true: Ground‑truth labels (integer‑encoded).
        y_pred: Model predictions (integer‑encoded).
        split:  Prefix for the PNG file (e.g. "train", "val", "test").
        label_encoder_path: Path to the pickled LabelEncoder.

    Saves:
        <split>_confusion_matrix.png in the current working directory.
    """
    # ----- Recover class names -------------------------------------------
    le: LabelEncoder = load_label_encoder(label_encoder_path)
    class_names = le.classes_
    n_classes = len(class_names)

    # ----- Compute confusion matrix --------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

    # ----- Plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.matshow(cm, cmap="plasma")
    fig.colorbar(im)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="left")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Write counts in each cell
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="red")

    plt.tight_layout()
    plt.savefig(f"{split}_confusion_matrix.png")
    plt.close()
