import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
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
    ):
        self.x_test_path = Path(x_test_path)
        self.y_test_path = Path(y_test_path)
        self.label_encoder_path = Path(label_encoder_path)
        self.predictions_path = Path(predictions_path)
        self.feature_importance_path = Path(feature_importance_path)
        self.confusion_matrix_path = Path(confusion_matrix_path)
        self.classification_report_path = Path(classification_report_path)

        # Load test data
        self.X_test = pd.read_csv(self.x_test_path)
        self.y_test = pd.read_csv(self.y_test_path).values.ravel()

        # Load label encoder
        if self.label_encoder_path.exists():
            self.label_encoder = joblib.load(self.label_encoder_path)
            logger.info(f"Loaded label encoder from {self.label_encoder_path}")
        else:
            logger.error(f"Label encoder not found at {self.label_encoder_path}")
            raise FileNotFoundError(f"{self.label_encoder_path} not found")

        # Load best pipeline (includes scaler + model)
        model_path = self.model_path
        if model_path.exists():
            self.pipeline = joblib.load(model_path)
            logger.info(f"Loaded best model pipeline from {model_path}")
        else:
            logger.error(f"Best model pipeline not found at {model_path}")
            raise FileNotFoundError(f"{model_path} not found")

    def evaluate(self):
        # Predict
        y_pred = self.pipeline.predict(self.X_test)

        # Decode labels if encoded
        if hasattr(self.label_encoder, "inverse_transform"):
            y_true_decoded = self.label_encoder.inverse_transform(self.y_test)
            y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        else:
            y_true_decoded = self.y_test
            y_pred_decoded = y_pred

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true_decoded, y_pred_decoded),
            "precision": precision_score(
                y_true_decoded, y_pred_decoded, average="macro", zero_division=0
            ),
            "recall": recall_score(
                y_true_decoded, y_pred_decoded, average="macro", zero_division=0
            ),
            "f1_score": f1_score(
                y_true_decoded, y_pred_decoded, average="macro", zero_division=0
            ),
        }

        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.model_metrics_path, index=False)
        logger.info(f"Evaluation metrics saved to {self.model_metrics_path}")

        # Plot multiclass ROC curve
        self.plot_multiclass_roc(self.y_test, self.pipeline.predict_proba(self.X_test))

        # Plot feature importance
        self.plot_feature_importance()

        return metrics

    def plot_multiclass_roc(self, y_true, y_proba):
        classes = self.label_encoder.classes_
        n_classes = len(classes)
        y_true_onehot = pd.get_dummies(self.label_encoder.inverse_transform(y_true))

        plt.figure(figsize=(8, 6))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_onehot.iloc[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                lw=2,
                label=f"ROC curve of class {classes[i]} (area = {roc_auc:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multiclass ROC Curve")
        plt.legend(loc="lower right")

        roc_path = self.output_dir / "multiclass_roc_curve.png"
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"Multiclass ROC curve saved to {roc_path}")

    def plot_feature_importance(self):
        # Extract model from pipeline
        model = self.pipeline.named_steps["clf"]

        # Check if model has feature_importances_ attribute
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = self.X_test.columns

            imp_df = pd.DataFrame(
                {
                    "Feature Name": feature_names,
                    "Importance": importances,
                }
            ).sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(12, 7))
            # Fix for seaborn future warning: add dummy hue and disable legend
            sns.barplot(
                x="Importance",
                y="Feature Name",
                data=imp_df,
                palette="plasma",
                hue=imp_df["Feature Name"],  # dummy hue to avoid warning
                legend=False,
            )
            plt.title(
                "Feature Importance in the Model Prediction",
                fontweight="black",
                size=20,
                pad=20,
            )
            plt.yticks(size=12)
            plt.xlabel("Importance")
            plt.ylabel("Feature Name")

            feat_imp_path = self.output_dir / "feature_importance.png"
            plt.savefig(feat_imp_path)
            plt.close()
            logger.info(f"Feature importance plot saved to {feat_imp_path}")
        else:
            logger.warning(
                "Model does not have feature_importances_ attribute; skipping feature importance plot."
            )


def main():
    try:
        params_path = Path("params.yaml")
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)

        model_path = params["artifacts"]["model_path"]
        scaler_path = params["artifacts"]["scaler_path"]
        x_test_path = params["data"]["x_test_path"]
        y_test_path = params["data"]["y_test_path"]
        label_encoder_path = params["artifacts"]["label_encoder_path"]
        model_metrics_path = params["reports"]["model_metrics_path"]
        predictions_path = params["reports"]["predictions_path"]
        feature_importance_path = params["reports"]["feature_importance_path"]
        confusion_matrix_path = params["reports"]["confusion_matrix_path"]
        classification_report_path = params["reports"]["classification_report_path"]

        evaluator = WeatherDiseaseEvaluator(
            x_test_path=x_test_path,
            y_test_path=y_test_path,
            label_encoder_path=label_encoder_path,
            predictions_path=predictions_path,
            feature_importance_path=feature_importance_path,
            confusion_matrix_path=confusion_matrix_path,
            classification_report_path=classification_report_path,
        )
        metrics = evaluator.evaluate()
        logger.info(f"Evaluation complete. Metrics: {metrics}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
