import os
import random
from pathlib import Path

import mlflow
import numpy as np
import matplotlib.pyplot as plt
import yaml

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from numpy.typing import ArrayLike
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

from src.utils.plotoutputs import plot_confusion_matrix
from src import logger

# === Configuration ===
load_dotenv(Path("./.env"))
SEED = 1024
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")

with open(Path("./params.yaml"), "r") as f:
    params = yaml.safe_load(f)
    x_train_path = Path(params["data"]["x_train_path"])
    y_train_path = Path(params["data"]["y_train_path"])
    x_test_path = Path(params["data"]["x_test_path"])
    y_test_path = Path(params["data"]["y_test_path"])


def config_mlflow() -> None:
    if DAGSHUB_REPO_OWNER is None or DAGSHUB_REPO is None or os.getenv("DAGSHUB_TOKEN") is None:
        raise ValueError("DAGSHUB_REPO_OWNER, DAGSHUB_REPO, and DAGSHUB_TOKEN environment variables must be set.")
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_token is None:
        raise ValueError("DAGSHUB_TOKEN environment variable must be set.")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO}.mlflow")


def create_mlflow_experiment(experiment_name: str) -> None:
    config_mlflow()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        logger.info(f"Created experiment: {experiment_name}")
    else:
        logger.info(f"Experiment already exists: {experiment_name}")
    mlflow.set_experiment(experiment_name)


def _evaluate_and_log_model(model, model_name: str, best_params: dict,
                            x_train: ArrayLike, y_train: ArrayLike) -> str:
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=SEED)

    model.fit(x_train, y_train)

    train_preds = model.predict(x_train)
    val_preds = model.predict(x_val)
    train_probs = model.predict_proba(x_train)
    val_probs = model.predict_proba(x_val)

    is_multiclass = len(np.unique(y_train)) > 2
    average_type = "macro" if is_multiclass else "binary"

    with mlflow.start_run() as run:
        mlflow.log_params(best_params)

        # Validation metrics
        mlflow.log_metric("accuracy", float(accuracy_score(y_val, val_preds)))
        mlflow.log_metric("f1", float(f1_score(y_val, val_preds, average=average_type)))
        mlflow.log_metric("precision", float(precision_score(y_val, val_preds, average=average_type)))
        mlflow.log_metric("recall", float(recall_score(y_val, val_preds, average=average_type)))
        if is_multiclass:
            mlflow.log_metric("roc_auc", float(roc_auc_score(y_val, val_probs, multi_class="ovr")))
        else:
            mlflow.log_metric("roc_auc", float(roc_auc_score(y_val, val_probs[:, 1])))

        # Training metrics
        mlflow.log_metric("train_accuracy", float(accuracy_score(y_train, train_preds)))
        mlflow.log_metric("train_f1", float(f1_score(y_train, train_preds, average=average_type)))
        mlflow.log_metric("train_precision", float(precision_score(y_train, train_preds, average=average_type)))
        mlflow.log_metric("train_recall", float(recall_score(y_train, train_preds, average=average_type)))
        if is_multiclass:
            mlflow.log_metric("train_roc_auc", float(roc_auc_score(y_train, train_probs, multi_class="ovr")))
        else:
            mlflow.log_metric("train_roc_auc", float(roc_auc_score(y_train, train_probs[:, 1])))

        # Save model
        if model_name == "LGBMClassifier":
            mlflow.lightgbm.log_model(model, "model") # type: ignore
        else:
            mlflow.sklearn.log_model(model, "model") # type: ignore

        # Confusion matrices
        plt.switch_backend("agg")
        plot_confusion_matrix(y_train, train_preds, "train")
        plot_confusion_matrix(y_val, val_preds, "val")
        mlflow.log_artifact("train_confusion_matrix.png")
        mlflow.log_artifact("val_confusion_matrix.png")
        os.remove("train_confusion_matrix.png")
        os.remove("val_confusion_matrix.png")

        # Params & Data artifacts
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("dvc.yaml")
        mlflow.log_artifact(str(x_train_path))
        mlflow.log_artifact(str(y_train_path))
        mlflow.log_artifact(str(x_test_path))
        mlflow.log_artifact(str(y_test_path))

        return run.info.run_id


def register_random_forest(x_train: ArrayLike, y_train: ArrayLike, best_params: dict) -> str:
    model = RandomForestClassifier(**best_params, random_state=SEED)
    return _evaluate_and_log_model(model, "RandomForest", best_params, x_train, y_train)


def register_gradient_boosting(x_train: ArrayLike, y_train: ArrayLike, best_params: dict) -> str:
    model = GradientBoostingClassifier(**best_params, random_state=SEED)
    return _evaluate_and_log_model(model, "GradientBoosting", best_params, x_train, y_train)


def register_hist_gradient_boosting(x_train: ArrayLike, y_train: ArrayLike, best_params: dict) -> str:
    model = HistGradientBoostingClassifier(**best_params, random_state=SEED)
    return _evaluate_and_log_model(model, "HistGradientBoosting", best_params, x_train, y_train)


def register_logistic_regression(x_train: ArrayLike, y_train: ArrayLike, best_params: dict) -> str:
    model = LogisticRegression(**best_params, random_state=SEED, max_iter=1000)
    return _evaluate_and_log_model(model, "LogisticRegression", best_params, x_train, y_train)


def register_lgbm_classifier(x_train: ArrayLike, y_train: ArrayLike, best_params: dict) -> str:
    model = LGBMClassifier(**best_params, random_state=SEED, verbose=-1)
    return _evaluate_and_log_model(model, "LGBMClassifier", best_params, x_train, y_train)


def load_model_by_name(model_name: str):
    config_mlflow()
    client = MlflowClient()
    registered_model = client.get_registered_model(model_name)
    if not registered_model.latest_versions:
        raise ValueError(f"No versions found for registered model '{model_name}'.")
    run_id = registered_model.latest_versions[-1].run_id
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
