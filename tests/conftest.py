"""Pytest configuration and shared fixtures for weather-health project."""

import sys
from pathlib import Path

import pytest

# Ensure src is in path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.parent))


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    import pandas as pd

    return pd.DataFrame(
        {
            "Age": [25, 30, 35],
            "Gender": ["Male", "Female", "Male"],
            "Temperature (C)": [37.0, 38.5, 36.5],
            "Humidity": [60, 70, 65],
            "prognosis": ["Flu", "Malaria", "Dengue"],
        }
    )


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "data": {
            "raw_data_path": "data/raw/test.csv",
            "interim_data_path": "data/interim/test.parquet",
            "x_train_path": "data/processed/x_train.csv",
            "y_train_path": "data/processed/y_train.csv",
            "x_test_path": "data/processed/x_test.csv",
            "y_test_path": "data/processed/y_test.csv",
            "output_dir": "data/processed",
            "target_col": "prognosis",
        },
        "modeling": {
            "n_trials": 2,
            "model_family": "logistic_regression",
            "objective_function": "Logloss",
            "loss_function": "F1",
        },
        "artifacts": {
            "model_path": "models/test_model.pkl",
            "scaler_path": "data/processed/test_scaler.pkl",
            "label_encoder_path": "data/processed/test_encoder.pkl",
        },
    }
