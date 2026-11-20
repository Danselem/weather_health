# model_trainer.py
"""
Model training module with Hydra configuration support.

Usage:
    python -m src.train                           # Default: dev + logistic_regression
    python -m src.train env=prod model=lightgbm  # Production with LightGBM
    python -m src.train model=random_forest n_trials=10
"""

import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from prefect import flow
from sklearn.model_selection import train_test_split

from src import logger
from src.config import load_config
from src.utils.mlflow_manager import (
    create_mlflow_experiment,
    load_model_by_name,
    register_best_model,
    register_gradient_boosting,
    register_hist_gradient_boosting,
    register_lgbm_classifier,
    register_logistic_regression,
    register_random_forest,
)
from src.utils.optimisation import classification_optimization

load_dotenv()

REGISTER_FUNCTIONS = {
    "random_forest": register_random_forest,
    "gradient_boosting": register_gradient_boosting,
    "logistic_regression": register_logistic_regression,
    "hist_gradient_boosting": register_hist_gradient_boosting,
    "lightgbm": register_lgbm_classifier,
}


class ModelTrainer:
    def __init__(self, config_path: str | None = None, overrides: list | None = None):
        self.config_path = config_path or self._get_config_dir()
        self.overrides = overrides or []
        self.cfg: DictConfig | None = None
        self.modeling_params: dict[str, Any] | None = None
        self.data_paths: dict[str, str] | None = None
        self.artifacts: dict[str, str] | None = None

        self._load_config()
        self._load_data()

    def _get_config_dir(self) -> Path:
        return Path(__file__).parent.parent / "config"

    def _load_config(self) -> None:
        self.cfg = load_config(self.config_path, self.overrides)

        self.modeling_params = self.cfg.modeling
        self.data_paths = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.artifacts = OmegaConf.to_container(self.cfg.artifacts, resolve=True)

        logger.info(f"Loaded config for environment: {self.cfg.env.name}")
        logger.info(f"Model: {self.modeling_params.model_family}")

    def _load_data(self) -> None:
        data_paths = self.data_paths
        self.x_train = pd.read_csv(Path(data_paths["x_train_path"]))
        self.y_train = pd.read_csv(Path(data_paths["y_train_path"])).values.ravel()
        self.x_test = pd.read_csv(Path(data_paths["x_test_path"]))
        self.y_test = pd.read_csv(Path(data_paths["y_test_path"])).values.ravel()

        split_cfg = self.cfg.get("data_split", {})
        validation_size = (
            split_cfg.get("validation_size", 0.15) if split_cfg else 0.15
        )
        random_state = split_cfg.get("random_state", 1024) if split_cfg else 1024
        self.x_fit, self.x_val, self.y_fit, self.y_val = train_test_split(
            self.x_train,
            self.y_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=self.y_train,
        )
        logger.info(
            "Train/validation/test sizes: %d/%d/%d (test set reserved for evaluate)",
            len(self.x_fit),
            len(self.x_val),
            len(self.x_test),
        )

    def _get_register_function(self, model_family: str):
        if model_family not in REGISTER_FUNCTIONS:
            raise ValueError(
                f"Unsupported model_family '{model_family}'. "
                f"Supported families: {list(REGISTER_FUNCTIONS.keys())}"
            )
        return REGISTER_FUNCTIONS[model_family]

    @flow(name="train_model", retries=3, retry_delay_seconds=10, log_prints=True)
    def run(self) -> None:
        model_family = self.modeling_params.model_family
        loss_function = self.modeling_params.loss_function
        n_trials = self.modeling_params.n_trials

        logger.info(f"Model Family: {model_family}")
        logger.info(f"Loss Function: {loss_function}")
        logger.info(f"Environment: {self.cfg.env.name}")

        create_mlflow_experiment(f"{model_family}_experiment")

        best_params = classification_optimization(
            x_train=self.x_fit,
            y_train=self.y_fit,
            model_family=model_family,
            loss_function=loss_function,
            num_trials=n_trials,
            diagnostic=True,
        )

        register_func = self._get_register_function(model_family)
        register_func(
            x_train=self.x_fit,
            y_train=self.y_fit,
            x_val=self.x_val,
            y_val=self.y_val,
            best_params=best_params,
        )

        register_best_model(model_family=model_family, loss_function=loss_function)

        # Persist the champion artifact so DVC has a real, reproducible output.
        model = load_model_by_name(f"{model_family}_best_model")
        model_path = Path(self.artifacts["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        logger.info(f"Champion model saved to {model_path}")

        logger.info("Training pipeline completed successfully.")

    def get_model_config(self) -> DictConfig:
        return self.cfg.model


@flow(name="train_pipeline")
def main(config_path: str | None = None, overrides: list | None = None) -> None:
    # If no overrides passed, try to parse from command line arguments
    if overrides is None:
        overrides = [
            arg
            for arg in sys.argv[1:]
            if arg.startswith(
                ("model=", "env=", "n_trials=", "loss_function=", "modeling.", "artifacts.", "data.")
            )
        ]

    trainer = ModelTrainer(config_path=config_path, overrides=overrides)
    trainer.run()


if __name__ == "__main__":
    main()
