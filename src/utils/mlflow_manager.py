import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from numpy.typing import ArrayLike
from prefect import task
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src import logger
from src.utils.plotoutputs import plot_confusion_matrix

# === Configuration ===
load_dotenv(Path("./.env"))
SEED = 1024
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")


def config_mlflow() -> None:
    """Configure MLflow; local SQLite is the default so the demo works offline."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    if "dagshub.com" in tracking_uri:
        if DAGSHUB_REPO_OWNER is None or os.getenv("DAGSHUB_TOKEN") is None:
            raise ValueError("DagsHub URL requires DAGSHUB_REPO_OWNER and DAGSHUB_TOKEN.")
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]


@task(
    name="create_mlflow_experiment", retries=3, retry_delay_seconds=10, log_prints=True
)
def create_mlflow_experiment(experiment_name: str) -> None:
    config_mlflow()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        logger.info(f"Created experiment: {experiment_name}")
    elif experiment.lifecycle_stage == "deleted":
        client = MlflowClient()
        client.restore_experiment(experiment.experiment_id)
        logger.info(f"Restored deleted experiment: {experiment_name}")
    else:
        logger.info(f"Experiment already exists: {experiment_name}")
    mlflow.set_experiment(experiment_name)


def _evaluate_and_log_model(
    model: Any,
    model_name: str,
    best_params: dict,
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    nested: bool = False,
) -> str:

    model = Pipeline([("scaler", MinMaxScaler()), ("clf", model)])
    model.fit(x_train, y_train)

    train_preds = model.predict(x_train)
    val_preds = model.predict(x_val)
    train_probs = model.predict_proba(x_train)
    val_probs = model.predict_proba(x_val)

    is_multiclass = len(np.unique(y_train)) > 2
    average_type = "macro" if is_multiclass else "binary"

    with mlflow.start_run(nested=nested) as run:
        mlflow.log_params(best_params)

        # Validation metrics
        mlflow.log_metric("accuracy", float(accuracy_score(y_val, val_preds)))
        mlflow.log_metric("f1", float(f1_score(y_val, val_preds, average=average_type)))
        mlflow.log_metric(
            "precision", float(precision_score(y_val, val_preds, average=average_type))
        )
        mlflow.log_metric(
            "recall", float(recall_score(y_val, val_preds, average=average_type))
        )
        if is_multiclass:
            mlflow.log_metric(
                "roc_auc", float(roc_auc_score(y_val, val_probs, multi_class="ovr"))
            )
        else:
            mlflow.log_metric("roc_auc", float(roc_auc_score(y_val, val_probs[:, 1])))

        # Training metrics
        mlflow.log_metric("train_accuracy", float(accuracy_score(y_train, train_preds)))
        mlflow.log_metric(
            "train_f1", float(f1_score(y_train, train_preds, average=average_type))
        )
        mlflow.log_metric(
            "train_precision",
            float(precision_score(y_train, train_preds, average=average_type)),
        )
        mlflow.log_metric(
            "train_recall",
            float(recall_score(y_train, train_preds, average=average_type)),
        )
        if is_multiclass:
            mlflow.log_metric(
                "train_roc_auc",
                float(roc_auc_score(y_train, train_probs, multi_class="ovr")),
            )
        else:
            mlflow.log_metric(
                "train_roc_auc", float(roc_auc_score(y_train, train_probs[:, 1]))
            )

        signature = infer_signature(x_val, val_preds)
        input_example = pd.DataFrame(x_val).iloc[:1]

        # The registered artifact is a complete preprocessing + model pipeline.
        mlflow.sklearn.log_model(
            model, "model", signature=signature, input_example=input_example
        )  # type: ignore

        # Confusion matrices
        plt.switch_backend("agg")
        plot_confusion_matrix(y_train, train_preds, "train")
        plot_confusion_matrix(y_val, val_preds, "val")
        mlflow.log_artifact("train_confusion_matrix.png")
        mlflow.log_artifact("val_confusion_matrix.png")
        os.remove("train_confusion_matrix.png")
        os.remove("val_confusion_matrix.png")

        # Params & Data artifacts
        if os.path.exists("params.yaml"):
            mlflow.log_artifact("params.yaml")
        if os.path.exists("dvc.yaml"):
            mlflow.log_artifact("dvc.yaml")

        return str(run.info.run_id)


@task(name="register_random_forest", retries=3, retry_delay_seconds=10, log_prints=True)
def register_random_forest(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    best_params: dict,
) -> str:
    """Register the Random Forest model and evaluate it on the validation set"""
    model = RandomForestClassifier(**best_params, random_state=SEED)
    return _evaluate_and_log_model(
        model, "RandomForest", best_params, x_train, y_train, x_val, y_val, nested=True
    )


@task(
    name="register_gradient_boosting",
    retries=3,
    retry_delay_seconds=10,
    log_prints=True,
)
def register_gradient_boosting(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    best_params: dict,
) -> str:
    """Register the Gradient Boosting model and evaluate it on the validation set"""
    model = GradientBoostingClassifier(**best_params, random_state=SEED)
    return _evaluate_and_log_model(
        model,
        "GradientBoosting",
        best_params,
        x_train,
        y_train,
        x_val,
        y_val,
        nested=True,
    )


@task(
    name="register_hist_gradient_boosting",
    retries=3,
    retry_delay_seconds=10,
    log_prints=True,
)
def register_hist_gradient_boosting(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    best_params: dict,
) -> str:
    """Register the Histogram Gradient Boosting model and evaluate it on the validation set"""
    model = HistGradientBoostingClassifier(**best_params, random_state=SEED)
    return _evaluate_and_log_model(
        model,
        "HistGradientBoosting",
        best_params,
        x_train,
        y_train,
        x_val,
        y_val,
        nested=True,
    )


@task(
    name="register_logistic_regression",
    retries=3,
    retry_delay_seconds=10,
    log_prints=True,
)
def register_logistic_regression(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    best_params: dict,
) -> str:
    """Register the Logistic Regression model and evaluate it on the validation set"""
    model = LogisticRegression(**best_params, random_state=SEED, max_iter=1000)
    return _evaluate_and_log_model(
        model,
        "LogisticRegression",
        best_params,
        x_train,
        y_train,
        x_val,
        y_val,
        nested=True,
    )


@task(
    name="register_lgbm_classifier", retries=3, retry_delay_seconds=10, log_prints=True
)
def register_lgbm_classifier(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    best_params: dict,
) -> str:
    """Register the LightGBM Classifier model and evaluate it on the validation set"""
    model = LGBMClassifier(**best_params, random_state=SEED, verbose=-1)
    return _evaluate_and_log_model(
        model,
        "LGBMClassifier",
        best_params,
        x_train,
        y_train,
        x_val,
        y_val,
        nested=True,
    )


@task(name="register_best_model", retries=3, retry_delay_seconds=10, log_prints=True)
def register_best_model(model_family: str, loss_function: str) -> None:
    """Register the best model after the optimization process with tags and description"""
    experiment_name = f"{model_family}_experiment"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.loss ASC"],
    )
    # The optimization parent run has no loss metric; only trial runs do.
    runs = [run for run in all_runs if run.data.metrics.get("loss") is not None]

    if not runs:
        raise ValueError("No runs with a 'loss' metric found for registration")

    logger.info(f"Found {len(runs)} candidate runs in experiment '{experiment.name}'")
    logger.info(f"Selected run loss={runs[0].data.metrics['loss']:.4f} (lower is better)")

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_name = f"{model_family}_best_model"

    try:
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info(f"Model registered: {result.name}, version: {result.version}")

        # Add tags
        client.set_model_version_tag(
            name=model_name,
            version=result.version,
            key="model_family",
            value=model_family,
        )
        client.set_model_version_tag(
            name=model_name,
            version=result.version,
            key="loss_function",
            value=loss_function,
        )

        # Set description
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=(
                f"Best {model_family} model optimized for {loss_function.lower()} "
                f"using Hyperopt. Registered from run {run_id}."
            ),
        )

        # A single mutable pointer for deployment; never use "latest version".
        client.set_registered_model_alias(
            name=model_name, alias="champion", version=result.version
        )
        logger.info(f"Promoted {model_name} v{result.version} to @champion")
        logger.info(f"Tags and description added to model version {result.version}")

    except Exception as e:
        logger.error(f"Failed to register or tag model: {e}")
        raise


@task(name="load_model_by_name", retries=3, retry_delay_seconds=10, log_prints=True)
def load_model_by_name(model_name: str) -> Any:
    config_mlflow()
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(model_name, "champion")
        model_uri = f"models:/{model_name}@champion"
    except Exception:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for registered model '{model_name}'.")
        latest = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{model_name}/{latest.version}"
    logger.info(f"Loading model from {model_uri}")
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Older registered artifacts were logged with the pyfunc flavor.
        return mlflow.pyfunc.load_model(model_uri)
