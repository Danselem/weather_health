"""Evaluation smoke test against the committed DVC artifacts."""

import yaml

from src.evaluate import WeatherDiseaseEvaluator


def test_evaluate_writes_metrics(tmp_path):
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    evaluator = WeatherDiseaseEvaluator(
        x_test_path=params["data"]["x_test_path"],
        y_test_path=params["data"]["y_test_path"],
        label_encoder_path=params["artifacts"]["label_encoder_path"],
        predictions_path=str(tmp_path / "predictions.csv"),
        feature_importance_path=str(tmp_path / "feature_importance.png"),
        confusion_matrix_path=str(tmp_path / "confusion_matrix.png"),
        classification_report_path=str(tmp_path / "classification_report.csv"),
        model_path=params["artifacts"]["model_path"],
        reference_path=str(tmp_path / "reference.parquet"),
    )
    evaluator.model_metrics_path = tmp_path / "model_metrics.csv"
    evaluator.evaluation_metrics_path = tmp_path / "evaluation_metrics.csv"

    metrics = evaluator.evaluate()

    assert metrics["accuracy"] > 0.90
    assert metrics["f1_score"] > 0.90
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "classification_report.csv").exists()
