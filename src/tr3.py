import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from pathlib import Path
from prefect import flow, task

from src.utils.mlflow_manager import (
    create_mlflow_experiment,
    register_random_forest,
    register_gradient_boosting,
    register_hist_gradient_boosting,
    register_logistic_regression,
    register_lgbm_classifier,
    register_best_model
)
from src.utils.optimisation import classification_optimization
from src import logger


# Mapping of model family to registration function
REGISTER_FUNCTIONS = {
    "random_forest": register_random_forest,
    "gradient_boosting": register_gradient_boosting,
    "logistic_regression": register_logistic_regression,
    "hist_gradient_boosting": register_hist_gradient_boosting,
    "lightgbm": register_lgbm_classifier
}

@flow(name="train_model", retries=3, retry_delay_seconds=10, log_prints=True)
def main():
    """Main function to run the optimization process."""
    load_dotenv()

    # Load configuration
    params_file = Path("params.yaml")
    config = yaml.safe_load(open(params_file, encoding="utf-8"))
    modeling_params = config["modeling"]
    data_paths = config["data"]
    artifacts = config.get("artifacts", {})

    n_trials = modeling_params["n_trials"]
    selected_loss_function = modeling_params["loss_function"]
    selected_model_family = modeling_params["model_family"]
    selected_objective_function = modeling_params["objective_function"]

    # Load data
    
    xtrain = pd.read_csv(Path(data_paths["x_train_path"]))
    ytrain = pd.read_csv(Path(data_paths["y_train_path"])).values.ravel()

    logger.info(f"Loaded training data from {data_paths['x_train_path']} and {data_paths['y_train_path']}")
    logger.info(f"Selected model family: {selected_model_family}")
    logger.info(f"Selected loss function: {selected_loss_function}")

    # Create MLflow experiment
    create_mlflow_experiment(f"{selected_model_family}_experiment")

    if selected_model_family not in REGISTER_FUNCTIONS:
        raise ValueError(f"Unsupported model_family '{selected_model_family}'. "
                         f"Supported families: {list(REGISTER_FUNCTIONS.keys())}")

    # Run optimization
    best_classification_params = classification_optimization(
        x_train=xtrain,
        y_train=ytrain,
        model_family=selected_model_family,
        loss_function=selected_loss_function,
        # objective_function=selected_objective_function,
        num_trials=n_trials,
        diagnostic=True
    )

    # Register model
    register_func = REGISTER_FUNCTIONS[selected_model_family]
    register_func(
        x_train=xtrain,
        y_train=ytrain,
        best_params=best_classification_params
    )

    # Register best model
    register_best_model(
        model_family=selected_model_family,
        loss_function=selected_loss_function
    )


if __name__ == "__main__":
    main()
