# model_trainer.py

from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from prefect import flow

from src import logger
from src.utils.mlflow_manager import (
    create_mlflow_experiment,
    register_best_model,
    register_gradient_boosting,
    register_hist_gradient_boosting,
    register_lgbm_classifier,
    register_logistic_regression,
    register_random_forest,
)
from src.utils.optimisation import classification_optimization

REGISTER_FUNCTIONS = {
    "random_forest": register_random_forest,
    "gradient_boosting": register_gradient_boosting,
    "logistic_regression": register_logistic_regression,
    "hist_gradient_boosting": register_hist_gradient_boosting,
    "lightgbm": register_lgbm_classifier,
}


class ModelTrainer:
    def __init__(self, config_path: str = "params.yaml"):
        self.config_path = config_path
        self.config = None
        self.modeling_params = None
        self.data_paths = None
        self.artifacts = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self._load_config()
        self._load_data()

    def _load_config(self):
        load_dotenv()
        with open(self.config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.modeling_params = self.config["modeling"]
        self.data_paths = self.config["data"]
        self.artifacts = self.config.get("artifacts", {})

    def _load_data(self):
        self.x_train = pd.read_csv(Path(self.data_paths["x_train_path"]))
        self.y_train = pd.read_csv(Path(self.data_paths["y_train_path"])).values.ravel()
        self.x_test = pd.read_csv(Path(self.data_paths["x_test_path"]))
        self.y_test = pd.read_csv(Path(self.data_paths["y_test_path"])).values.ravel()
        logger.info("Training and test data loaded successfully.")

    def _get_register_function(self, model_family: str):
        if model_family not in REGISTER_FUNCTIONS:
            raise ValueError(
                f"Unsupported model_family '{model_family}'. "
                f"Supported families: {list(REGISTER_FUNCTIONS.keys())}"
            )
        return REGISTER_FUNCTIONS[model_family]

    @flow(name="train_model", retries=3, retry_delay_seconds=10, log_prints=True)
    def run(self):
        model_family = self.modeling_params["model_family"]
        loss_function = self.modeling_params["loss_function"]
        n_trials = self.modeling_params["n_trials"]

        logger.info(f"Model Family: {model_family}")
        logger.info(f"Loss Function: {loss_function}")
        create_mlflow_experiment(f"{model_family}_experiment")

        # Optimization
        best_params = classification_optimization(
            x_train=self.x_train,
            y_train=self.y_train,
            model_family=model_family,
            loss_function=loss_function,
            num_trials=n_trials,
            diagnostic=True,
        )

        # Model registration
        register_func = self._get_register_function(model_family)
        register_func(
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_test,
            y_val=self.y_test,
            best_params=best_params,
        )

        register_best_model(model_family=model_family, loss_function=loss_function)
        logger.info("Training pipeline completed successfully.")


# Optional entry point
if __name__ == "__main__":
    params_file = "params.yaml"
    trainer = ModelTrainer(params_file)
    trainer.run()
