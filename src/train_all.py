# train_all.py
"""
Multi-model training for staging/production environments.
Trains all model families and registers the best one.

Usage:
    python -m src.train_all env=prod
"""

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
    register_best_model,
    register_gradient_boosting,
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
    "lightgbm": register_lgbm_classifier,
}

MODEL_FAMILIES = ["lightgbm", "random_forest", "gradient_boosting", "logistic_regression"]


class MultiModelTrainer:
    def __init__(self, config_path: str | None = None, overrides: list | None = None):
        self.config_path = config_path or self._get_config_dir()
        self.overrides = overrides or []
        self.cfg: DictConfig | None = None
        self.modeling_params: dict[str, Any] | None = None
        self.data_paths: dict[str, str] | None = None

        self._load_config()
        self._load_data()

    def _get_config_dir(self) -> Path:
        return Path(__file__).parent.parent / "config"

    def _load_config(self) -> None:
        self.cfg = load_config(self.config_path, self.overrides)

        self.modeling_params = self.cfg.modeling

        env_name = self.cfg.env.name
        if env_name == "staging":
            self.modeling_params.n_trials = 3
        elif env_name == "prod":
            self.modeling_params.n_trials = 10

    def _load_data(self) -> None:
        data_paths = OmegaConf.to_container(self.cfg.data, resolve=True)
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
            "Train/validation/test sizes: %d/%d/%d",
            len(self.x_fit),
            len(self.x_val),
            len(self.x_test),
        )

    def _train_single_model(self, model_family: str) -> dict[str, Any]:
        loss_function = self.modeling_params.loss_function
        n_trials = self.modeling_params.n_trials

        logger.info(f"Training {model_family} with {n_trials} trials")

        create_mlflow_experiment(f"{model_family}_experiment")

        best_params = classification_optimization(
            x_train=self.x_fit,
            y_train=self.y_fit,
            model_family=model_family,
            loss_function=loss_function,
            num_trials=n_trials,
            diagnostic=True,
        )

        register_func = REGISTER_FUNCTIONS[model_family]
        register_func(
            x_train=self.x_fit,
            y_train=self.y_fit,
            x_val=self.x_val,
            y_val=self.y_val,
            best_params=best_params,
        )

        register_best_model(model_family=model_family, loss_function=loss_function)

        return {"model_family": model_family, "best_params": best_params}

    @flow(name="train_all_models", retries=3, retry_delay_seconds=10, log_prints=True)
    def run(self) -> None:
        env_name = self.cfg.env.name
        logger.info(f"Multi-model training for environment: {env_name}")
        logger.info(f"Training {len(MODEL_FAMILIES)} models: {MODEL_FAMILIES}")

        results = {}
        for model_family in MODEL_FAMILIES:
            try:
                result = self._train_single_model(model_family)
                results[model_family] = result
                logger.info(f"Completed {model_family}")
            except Exception as e:
                logger.error(f"Failed to train {model_family}: {e}")
                results[model_family] = {"error": str(e)}

        logger.info(f"Training results: {results}")
        logger.info("Multi-model training pipeline completed successfully.")


def main(config_path: str | None = None, overrides: list | None = None) -> None:
    if overrides is None:
        overrides = [
            arg
            for arg in sys.argv[1:]
            if arg.startswith(
                ("model=", "env=", "n_trials=", "loss_function=", "modeling.", "artifacts.", "data.")
            )
        ]

    trainer = MultiModelTrainer(config_path=config_path, overrides=overrides)
    trainer.run()


if __name__ == "__main__":
    main()
