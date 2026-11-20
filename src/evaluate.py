"""Evaluate the champion model and write the artifacts declared in dvc.yaml."""

import logging
import pickle
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherDiseaseEvaluator:
    def __init__(
        self,
        x_test_path: str,
        y_test_path: str,
        label_encoder_path: str,
        predictions_path: str,
        feature_importance_path: str,
        confusion_matrix_path: str,
        classification_report_path: str,
        model_path: str | None = None,
    ):
        self.x_test_path = Path(x_test_path)
        self.y_test_path = Path(y_test_path)
        self.label_encoder_path = Path(label_encoder_path)
        self.predictions_path = Path(predictions_path)
        self.feature_importance_path = Path(feature_importance_path)
        self.confusion_matrix_path = Path(confusion_matrix_path)
        self.classification_report_path = Path(classification_report_path)
        self.model_path = Path(model_path or "models/weather_disease_model.pkl")
        self.model_metrics_path = Path("reports/model_metrics.csv")
        self.evaluation_metrics_path = Path("reports/evaluation_metrics.csv")
        self.output_dir = Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.X_test = pd.read_csv(self.x_test_path)
        self.y_test = pd.read_csv(self.y_test_path).values.ravel()

        self.label_encoder = self._load(self.label_encoder_path)
        self.pipeline = self._load(self.model_path)

    @staticmethod
    def _load(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as file:
                return pickle.load(file)

    def evaluate(self) -> dict[str, float]:
        y_pred = self.pipeline.predict(self.X_test)
        y_pred = pd.Series(y_pred).to_numpy().ravel()
        y_true = self.label_encoder.inverse_transform(self.y_test)
        y_pred = self.label_encoder.inverse_transform(y_pred)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.model_metrics_path, index=False)
        metrics_df.to_csv(self.evaluation_metrics_path, index=False)
        logger.info("Metrics saved to %s", self.evaluation_metrics_path)

        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
            self.predictions_path, index=False
        )

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(self.classification_report_path)
        logger.info("Classification report saved to %s", self.classification_report_path)

        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_feature_importance()
        return metrics

    def plot_confusion_matrix(self, y_true: Any, y_pred: Any) -> None:
        labels = list(self.label_encoder.classes_)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.confusion_matrix_path)
        plt.close()
        logger.info("Confusion matrix saved to %s", self.confusion_matrix_path)

    def plot_feature_importance(self) -> None:
        model = self.pipeline
        if hasattr(model, "named_steps"):
            model = model.named_steps.get("clf", model)

        if not hasattr(model, "feature_importances_"):
            logger.warning("Artifact has no feature_importances_; writing placeholder")
            plt.figure(figsize=(8, 4))
            plt.text(
                0.5,
                0.5,
                "Feature importance unavailable\nfor this artifact type",
                ha="center",
                va="center",
            )
            plt.axis("off")
            plt.savefig(self.feature_importance_path)
            plt.close()
            return

        importances = pd.DataFrame(
            {
                "Feature Name": self.X_test.columns,
                "Importance": model.feature_importances_,
            }
        ).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(12, 7))
        sns.barplot(
            x="Importance",
            y="Feature Name",
            data=importances,
            hue="Feature Name",
            palette="plasma",
            legend=False,
        )
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(self.feature_importance_path)
        plt.close()
        logger.info("Feature importance saved to %s", self.feature_importance_path)


def main() -> None:
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    evaluator = WeatherDiseaseEvaluator(
        x_test_path=params["data"]["x_test_path"],
        y_test_path=params["data"]["y_test_path"],
        label_encoder_path=params["artifacts"]["label_encoder_path"],
        predictions_path=params["reports"]["predictions_path"],
        feature_importance_path=params["reports"]["feature_importance_path"],
        confusion_matrix_path=params["reports"]["confusion_matrix_path"],
        classification_report_path=params["reports"]["classification_report_path"],
        model_path=params["artifacts"]["model_path"],
    )
    metrics = evaluator.evaluate()
    logger.info("Evaluation complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
