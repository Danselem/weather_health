"""Unit tests for utility functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestOptimisation:
    """Tests for the classification_optimization function."""

    def test_optimization_returns_params(self):
        """Test that optimization returns a dictionary of parameters."""
        from src.utils.optimisation import classification_optimization

        X_train = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "feature2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            }
        )
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        result = classification_optimization(
            x_train=X_train,
            y_train=y_train,
            model_family="logistic_regression",
            loss_function="F1",
            num_trials=2,
            diagnostic=False,
        )

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_unsupported_model_family(self):
        """Test that unsupported model family raises ValueError."""
        from src.utils.optimisation import classification_optimization

        X_train = pd.DataFrame({"feature1": [1.0, 2.0, 3.0]})
        y_train = np.array([0, 1, 0])

        with pytest.raises(ValueError, match="Unsupported model_family"):
            classification_optimization(
                x_train=X_train,
                y_train=y_train,
                model_family="unsupported_model",
                loss_function="F1",
                num_trials=1,
                diagnostic=False,
            )


class TestPlotOutputs:
    """Tests for plotting utility functions."""

    def test_plot_confusion_matrix_runs(self):
        """Test that plot_confusion_matrix executes without error."""
        from src.utils.plotoutputs import plot_confusion_matrix

        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        plot_confusion_matrix(y_true, y_pred, "test")