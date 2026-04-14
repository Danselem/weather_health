"""Script to download model artifact and save it to the local filesystem.
This is used by the Dockerfile to build the image and deploy the model.

Usage:
    python -m src.fetch_best_model                    # Fetch default (lightgbm)
    python -m src.fetch_best_model lightgbm     # Specify model family
"""

import pickle
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.utils.mlflow_manager import load_model_by_name


def main(model_family: str = "lightgbm"):
    """Fetch best model from MLflow and save to local filesystem."""
    load_dotenv()

    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    model_path = Path(params["artifacts"]["model_path"])
    model_repo = model_path.parent
    model_repo.mkdir(parents=True, exist_ok=True)

    print(f"Fetching best model: {model_family}_best_model")
    model = load_model_by_name(f"{model_family}_best_model")

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    params["modeling"]["model_family"] = model_family
    yaml.safe_dump(params, open("params.yaml", "w"))

    print(f"Model saved to {model_path}")
    print(f"Updated params.yaml with model_family: {model_family}")


if __name__ == "__main__":
    model_family = sys.argv[1] if len(sys.argv) > 1 else "lightgbm"
    main(model_family)
