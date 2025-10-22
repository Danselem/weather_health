from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src import logger
from utils.mlflow_manager import (
    create_mlflow_experiment,
    register_best_model,
    register_gradient_boosting,
    register_hist_gradient_boosting,
    register_lgbm_classifier,
    register_logistic_regression,
    register_random_forest,
)
from utils.optimisation import classification_optimization


def main():
    """Main function to run the optimization process. It loads the training
    dataset, optimizes the hyperparameters, registers the best experiment,
    and registers the best model."""
    load_dotenv()

    # Load params.yaml file
    params_file = Path("params.yaml")
    modeling_params = yaml.safe_load(open(params_file, encoding="utf-8"))["modeling"]
    n_trials = modeling_params["n_trials"]
    selected_loss_function = modeling_params["loss_function"]
    selected_model_family = modeling_params["model_family"]
    selected_objective_function = modeling_params["objective_function"]

    scaler_path = Path(params_file["artifacts"]["scaler_path"])
    x_train_path = Path(params_file["data"]["x_train_path"])
    xtrain = pd.read_csv(x_train_path)
    y_train_path = Path(params_file["data"]["y_train_path"])
    ytrain = pd.read_csv(y_train_path).values.ravel()
    x_test_path = Path(params_file["data"]["x_test_path"])
    xtest = pd.read_csv(x_test_path)
    y_test_path = Path(params_file["data"]["y_test_path"])
    x_test = pd.read_csv(y_test_path)
    y_test = pd.read_csv(y_test_path)
    logger.info(f"Loaded training data from {x_train_path} and {y_train_path}")

    create_mlflow_experiment(f"{selected_model_family}_experiment")

    # Load the appropriate training dataset based on the model family
    if selected_model_family == "random_forest":

        best_classification_params = classification_optimization(
            x_train=xtrain,
            y_train=ytrain,
            model_family=selected_model_family,
            loss_function=selected_loss_function,
            objective_function=selected_objective_function,
            num_trials=n_trials,
            diagnostic=True,
        )

        register_random_forest(
            x_train=xtrain,
            y_train=ytrain,
            loss_function=selected_loss_function,
            best_params=best_classification_params,
        )

        register_best_model(
            model_family=selected_model_family, loss_function=selected_loss_function
        )

    elif selected_model_family == "gradient_boosting":

        best_classification_params = classification_optimization(
            x_train=xtrain,
            y_train=ytrain,
            model_family=selected_model_family,
            loss_function=selected_loss_function,
            objective_function=selected_objective_function,
            num_trials=n_trials,
            diagnostic=True,
        )

        register_gradient_boosting(
            x_train=xtrain, y_train=ytrain, best_params=best_classification_params
        )

        register_best_model(
            model_family=selected_model_family, loss_function=selected_loss_function
        )

    elif selected_model_family == "logistic_regression":

        best_classification_params = classification_optimization(
            x_train=xtrain,
            y_train=ytrain,
            model_family=selected_model_family,
            loss_function=selected_loss_function,
            objective_function=selected_objective_function,
            num_trials=n_trials,
            diagnostic=True,
        )

        register_logistic_regression(
            x_train=xtrain, y_train=ytrain, best_params=best_classification_params
        )

        register_best_model(
            model_family=selected_model_family, loss_function=selected_loss_function
        )

    elif selected_model_family == "hist_gradient_boosting":
        best_classification_params = classification_optimization(
            x_train=xtrain,
            y_train=ytrain,
            model_family=selected_model_family,
            loss_function=selected_loss_function,
            objective_function=selected_objective_function,
            num_trials=n_trials,
            diagnostic=True,
        )

        register_hist_gradient_boosting(
            x_train=xtrain, y_train=ytrain, best_params=best_classification_params
        )

        register_best_model(
            model_family=selected_model_family, loss_function=selected_loss_function
        )
    elif selected_model_family == "lightgbm":
        best_classification_params = classification_optimization(
            x_train=xtrain,
            y_train=ytrain,
            model_family=selected_model_family,
            loss_function=selected_loss_function,
            objective_function=selected_objective_function,
            num_trials=n_trials,
            diagnostic=True,
        )

        register_lgbm_classifier(
            x_train=xtrain, y_train=ytrain, best_params=best_classification_params
        )

        register_best_model(
            model_family=selected_model_family, loss_function=selected_loss_function
        )

    else:
        raise ValueError(
            f"Unsupported model_family '{selected_model_family}'. "
            "Supported families are 'catboost', 'xgboost', and 'random_forest'."
        )


if __name__ == "__main__":
    main()
