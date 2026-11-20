"""Calibration metric tests."""

import numpy as np

from src.calibration import expected_calibration_error, multiclass_brier


def test_multiclass_brier_perfect_predictions():
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    y_true = np.array([0, 1, 0])
    assert multiclass_brier(y_true, probs) > 0
    perfect = np.eye(2)[y_true]
    assert multiclass_brier(y_true, perfect) == 0.0


def test_ece_perfectly_calibrated_bins():
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    y_true = np.array([0, 0, 0, 0])
    ece, bins = expected_calibration_error(y_true, probs, n_bins=2)
    assert ece == 0.0
    assert len(bins) == 1
