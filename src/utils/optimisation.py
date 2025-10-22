from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from numpy.typing import ArrayLike
from prefect import task
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split

from src import logger

seed = 1024


@task(
    name="classification_objective", retries=3, retry_delay_seconds=10, log_prints=True
)
def classification_objective(
    x_train: pd.DataFrame,
    y_train: ArrayLike,
    model_family: str,
    loss_function: Literal["Accuracy", "F1", "Precision"],
    params: dict,
) -> dict:
    """Objective function used by Hyperopt."""

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed
    )

    if model_family == "random_forest":
        model = RandomForestClassifier(**params)
    elif model_family == "gradient_boosting":
        model = GradientBoostingClassifier(**params)
    elif model_family == "logistic_regression":
        model = LogisticRegression(**params)
    elif model_family == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(**params)
    elif model_family == "lightgbm":
        model = LGBMClassifier(**params)
    else:
        raise ValueError(f"Unsupported model_family '{model_family}'")

    with mlflow.start_run(nested=True):  # nested=True
        mlflow.log_params(params)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)

        signature = infer_signature(x_val, y_pred)
        input_example = x_val.iloc[:1]

        y_pred = np.asarray(y_pred).ravel()
        if loss_function == "F1":
            loss = 1 - f1_score(y_val, y_pred, average="weighted")
        elif loss_function == "Accuracy":
            loss = 1 - accuracy_score(y_val, y_pred)
        elif loss_function == "Precision":
            loss = 1 - precision_score(y_val, y_pred, average="weighted")
        else:
            raise ValueError(f"Unsupported loss_function '{loss_function}'")

        # Log metrics
        mlflow.log_metric("loss", float(loss))
        mlflow.log_metric("accuracy", float(accuracy_score(y_val, y_pred)))
        mlflow.log_metric("f1", float(f1_score(y_val, y_pred, average="weighted")))
        mlflow.log_metric(
            "precision", float(precision_score(y_val, y_pred, average="weighted"))
        )

        #        # Log the model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )  # type: ignore

    return {"loss": loss, "status": STATUS_OK}


@task(name="hyperparameter_tuning", retries=3, retry_delay_seconds=10, log_prints=True)
def classification_optimization(
    x_train: ArrayLike,
    y_train: ArrayLike,
    model_family: str,
    loss_function: Literal["Accuracy", "F1", "Precision"],
    num_trials: int = 50,
    diagnostic: bool = False,
) -> dict:
    """Run hyperparameter optimization with Hyperopt."""

    if model_family == "random_forest":
        search_space = {
            "max_depth": scope.int(hp.quniform("max_depth", 3, 30, 1)),
            "n_estimators": scope.int(hp.quniform("n_estimators", 50, 300, 10)),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 4, 1)),
        }

    elif model_family == "gradient_boosting":
        search_space = {
            "n_estimators": scope.int(hp.quniform("n_estimators", 50, 300, 10)),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "max_depth": scope.int(hp.quniform("max_depth", 3, 10, 1)),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
        }

    elif model_family == "logistic_regression":
        search_space = {
            "C": hp.loguniform("C", np.log(0.001), np.log(10)),
            "solver": hp.choice(
                "solver",
                [
                    "liblinear",
                    "saga",
                ],
            ),
            "penalty": hp.choice(
                "penalty",
                [
                    "l2",
                    "l1",
                ],
            ),
            "max_iter": 200,
        }

    elif model_family == "hist_gradient_boosting":
        search_space = {
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "max_iter": scope.int(hp.quniform("max_iter", 100, 300, 10)),
            "max_leaf_nodes": scope.int(hp.quniform("max_leaf_nodes", 10, 50, 1)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 20, 1)),
            "l2_regularization": hp.loguniform(
                "l2_regularization", np.log(1e-4), np.log(1)
            ),
        }

    elif model_family == "lightgbm":
        search_space = {
            "n_estimators": scope.int(hp.quniform("n_estimators", 50, 300, 10)),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
            "num_leaves": scope.int(hp.quniform("num_leaves", 15, 150, 1)),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        }

    else:
        raise ValueError(f"Unsupported model_family '{model_family}'")

    trials = Trials()
    rstate = np.random.default_rng(seed)

    logger.info(f"Starting optimization for {model_family}...")

    # End any previous active run
    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.set_experiment(f"{model_family}_experiment")

    with mlflow.start_run(
        run_name=f"{model_family}_optimization",
    ):  # nested=True
        best_params = fmin(
            fn=lambda params: classification_objective(
                x_train, y_train, model_family, loss_function, params
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=trials,
            rstate=rstate,
        )

    # Post-process categorical choices
    if model_family == "logistic_regression" and best_params is not None:
        solver_list = [
            "liblinear",
            "saga",
        ]
        penalty_list = ["l2", "l1"]
        if best_params.get("solver") is not None:
            best_params["solver"] = solver_list[best_params["solver"]]
        if best_params.get("penalty") is not None:
            best_params["penalty"] = penalty_list[best_params["penalty"]]
        if best_params.get("C") is not None:
            best_params["C"] = float(best_params["C"])

    # Ensure correct casting
    if best_params is not None:
        for key in best_params:
            if isinstance(best_params[key], float) and key not in [
                "learning_rate",
                "C",
                "l2_regularization",
                "subsample",
                "colsample_bytree",
            ]:
                best_params[key] = int(best_params[key])

    logger.info(f"Best parameters for {model_family}: {best_params}")

    if diagnostic:
        for i, trial in enumerate(trials.trials):
            logger.debug(f"Trial #{i}: loss = {trial['result']['loss']}")

    return best_params if best_params is not None else {}
