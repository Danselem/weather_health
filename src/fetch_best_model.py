"""Fetch the registered champion and save it for the Docker build.

Usage:
    python -m src.fetch_best_model               # lightgbm
    python -m src.fetch_best_model lightgbm
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.utils.mlflow_manager import load_model_by_name


def _champion_metadata(model_name: str) -> dict[str, Any]:
    try:
        client = MlflowClient()
        version = client.get_model_version_by_alias(model_name, "champion")
        return {
            "model_name": model_name,
            "version": version.version,
            "run_id": version.run_id,
            "alias": "champion",
        }
    except Exception:
        return {"model_name": model_name, "version": "unknown", "alias": "champion"}


def main(model_family: str = "lightgbm") -> None:
    load_dotenv()

    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    model_path = Path(params["artifacts"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = f"{model_family}_best_model"
    print(f"Fetching registered champion: {model_name}@champion")
    model = load_model_by_name(model_name)

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    metadata = _champion_metadata(model_name)
    metadata_path = model_path.parent / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}: {metadata}")


if __name__ == "__main__":
    family = sys.argv[1] if len(sys.argv) > 1 else "lightgbm"
    main(family)
