"""Script to download model artifact and save it to the local filesystem.
This is used by the Dockerfile to build the image and deploy the model."""

import pickle
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.utils.mlflow_manager import load_model_by_name


def main():
    """
    By the params.yaml file, we know the model family to download.
    """

    load_dotenv()

    params_file = Path("params.yaml")
    params = yaml.safe_load(params_file.read_text())
    model_path = Path(params["artifacts"]["model_path"])
    model_repo = model_path.parent
    model_repo.mkdir(parents=True, exist_ok=True)

    # Load params.yaml file
    modeling_params = yaml.safe_load(open("params.yaml", encoding="utf-8"))["modeling"]
    model_family = modeling_params["model_family"]
    # Load the model
    model = load_model_by_name(f"{model_family}_best_model")
    # Save the model as pkl
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    main()
