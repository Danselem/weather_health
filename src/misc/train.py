from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import logger
from utils.mlflow_manager import *


class WeatherDiseaseTrainer:
    """
    Trains multiple classifiers on weather-disease data using Hyperopt for tuning,
    saves the best model and evaluation metrics. Applies saved MinMaxScaler during training.
    Uses train/validation/test split to avoid overfitting and ensure generalizability.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        x_train_path: str,
        y_train_path: str,
        x_test_path: str,
        y_test_path: str,
        model_metrics_path: str,
        predictions_path: str,
        random_state: int = 1024,
    ):
        self.model_path = Path(model_path)
        self.model_metrics_path = Path(model_metrics_path)
        self.predictions_path = Path(predictions_path)
        self.random_state = random_state

        # Load training data and split into train/validation
        X = pd.read_csv(x_train_path)
        y = pd.read_csv(y_train_path).values.ravel()

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Load test set
        self.X_test = pd.read_csv(x_test_path)
        self.y_test = pd.read_csv(y_test_path).values.ravel()

        # Load saved scaler
        self.scaler_path = Path(scaler_path)
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Loaded MinMaxScaler from {self.scaler_path}")
        else:
            logger.error(f"MinMaxScaler not found at {self.scaler_path}")
            raise FileNotFoundError(f"{self.scaler_path} not found")

        self.metrics_df = pd.DataFrame(
            columns=["accuracy", "precision", "recall", "f1_score", "model"]
        )

    def _evaluate(self, y_true, y_pred) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        }

    def _objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        model_type = params.pop("type")
        logger.info(f"Evaluating model: {model_type} with params: {params}")

        if model_type == "rf":
            model = RandomForestClassifier(**params, random_state=self.random_state)
        elif model_type == "lr":
            model = LogisticRegression(
                **params, random_state=self.random_state, max_iter=1000
            )
        elif model_type == "gb":
            model = GradientBoostingClassifier(**params, random_state=self.random_state)
        else:
            raise ValueError("Unsupported model type")

        pipeline = Pipeline([("scaler", self.scaler), ("clf", model)])

        pipeline.fit(self.X_train, self.y_train)
        preds = pipeline.predict(self.X_val)
        scores = self._evaluate(self.y_val, preds)

        self.metrics_df.loc[len(self.metrics_df)] = [
            scores["accuracy"],
            scores["precision"],
            scores["recall"],
            scores["f1_score"],
            model_type,
        ]

        return {"loss": -scores["f1_score"], "status": STATUS_OK, "model": pipeline}

    def run(self):
        logger.info("Starting hyperparameter optimization...")

        search_space = hp.choice(
            "classifier_type",
            [
                {
                    "type": "rf",
                    "n_estimators": hp.choice("rf_n_estimators", [50, 100, 200]),
                    "max_depth": hp.choice("rf_max_depth", [5, 10, 20, None]),
                },
                {
                    "type": "lr",
                    "C": hp.loguniform("lr_C", np.log(0.01), np.log(10)),
                    "solver": hp.choice("lr_solver", ["lbfgs", "liblinear"]),
                },
                {
                    "type": "gb",
                    "n_estimators": hp.choice("gb_n_estimators", [50, 100, 200]),
                    "learning_rate": hp.uniform("gb_learning_rate", 0.01, 0.3),
                    "max_depth": hp.choice("gb_max_depth", [3, 5, 7]),
                },
            ],
        )

        trials = Trials()
        best = fmin(
            fn=self._objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials,
        )

        # Save all model metrics from validation
        self.metrics_df.to_csv(self.model_metrics_path, index=False)
        logger.info(f"All model metrics saved to {self.model_metrics_path}")

        # Select best model from trials and evaluate on test set
        best_idx = np.argmin([t["result"]["loss"] for t in trials.trials])
        best_pipeline = trials.trials[best_idx]["result"]["model"]

        final_preds = best_pipeline.predict(self.X_test)
        final_scores = self._evaluate(self.y_test, final_preds)
        logger.info(f"Final model evaluation on test set: {final_scores}")

        # Save best model
        joblib.dump(best_pipeline, self.model_path)
        logger.info(f"Best model saved to {self.model_path}")

        # Save test predictions
        pd.DataFrame({"y_true": self.y_test, "y_pred": final_preds}).to_csv(
            self.predictions_path, index=False
        )
        logger.info(f"Predictions saved to {self.predictions_path}")


def main():
    params_path = Path("params.yaml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    scaler_path = params["artifacts"]["scaler_path"]
    x_train_path = params["data"]["x_train_path"]
    y_train_path = params["data"]["y_train_path"]
    x_test_path = params["data"]["x_test_path"]
    y_test_path = params["data"]["y_test_path"]

    model_path = params["artifacts"]["model_path"]
    model_metrics_path = params["reports"]["model_metrics_path"]
    predictions_path = params["reports"]["predictions_path"]

    try:
        trainer = WeatherDiseaseTrainer(
            model_path=model_path,
            scaler_path=scaler_path,
            x_train_path=x_train_path,
            y_train_path=y_train_path,
            x_test_path=x_test_path,
            y_test_path=y_test_path,
            model_metrics_path=model_metrics_path,
            predictions_path=predictions_path,
        )
        trainer.run()
    except Exception as e:
        logger.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()
