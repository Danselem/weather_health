"""Unit tests for WeatherDiseasePreprocessor class."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml


class TestWeatherDiseasePreprocessor:
    """Tests for the WeatherDiseasePreprocessor class."""

    @pytest.fixture
    def temp_parquet(self, test_data_dir):
        """Create a temporary parquet file for testing."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [0.5, 1.5, 2.5, 3.5, 4.5],
                "prognosis": ["Flu", "Malaria", "Dengue", "Flu", "Malaria"],
            }
        )
        parquet_path = test_data_dir / "test_input.parquet"
        df.to_parquet(parquet_path, index=False)
        return parquet_path

    @pytest.fixture
    def temp_config(self, temp_parquet, test_data_dir):
        """Create a temporary config dictionary for testing."""
        return {
            "data": {
                "interim_data_path": str(temp_parquet),
                "x_train_path": str(test_data_dir / "x_train.csv"),
                "y_train_path": str(test_data_dir / "y_train.csv"),
                "x_test_path": str(test_data_dir / "x_test.csv"),
                "y_test_path": str(test_data_dir / "y_test.csv"),
            },
            "artifacts": {
                "scaler_path": str(test_data_dir / "scaler.pkl"),
                "label_encoder_path": str(test_data_dir / "encoder.pkl"),
            },
        }

    def test_init(self, temp_config):
        """Test preprocessor initialization."""
        from src.transform import WeatherDiseasePreprocessor

        config_path = test_data_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(temp_config, f)

        processor = WeatherDiseasePreprocessor(config_path=str(config_path))

        assert processor.input_path == Path(temp_config["data"]["interim_data_path"])
        assert processor.scaler is not None
        assert processor.label_encoder is not None

    def test_load_data(self, temp_config):
        """Test loading data from parquet."""
        from src.transform import WeatherDiseasePreprocessor

        config_path = test_data_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(temp_config, f)

        processor = WeatherDiseasePreprocessor(config_path=str(config_path))
        data = processor.load_data()

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5
        assert "prognosis" in data.columns

    def test_split_data(self, temp_config):
        """Test splitting data into train and test."""
        from src.transform import WeatherDiseasePreprocessor

        config_path = test_data_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(temp_config, f)

        processor = WeatherDiseasePreprocessor(config_path=str(config_path))
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [0.5, 1.5, 2.5, 3.5, 4.5],
                "prognosis": ["Flu", "Malaria", "Dengue", "Flu", "Malaria"],
            }
        )

        train_df, test_df = processor.split_data(df)

        assert len(train_df) == 4
        assert len(test_df) == 1

    def test_encode_labels(self, temp_config):
        """Test encoding target labels."""
        from src.transform import WeatherDiseasePreprocessor

        config_path = test_data_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(temp_config, f)

        processor = WeatherDiseasePreprocessor(config_path=str(config_path))
        y = pd.Series(["Flu", "Malaria", "Dengue", "Flu", "Malaria"], name="prognosis")

        encoded = processor.encode_labels(y)

        assert len(encoded) == 5
        assert encoded.dtype in [int, "int64"]

    def test_scale_features(self, temp_config):
        """Test scaling features with MinMaxScaler."""
        from src.transform import WeatherDiseasePreprocessor

        config_path = test_data_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(temp_config, f)

        processor = WeatherDiseasePreprocessor(config_path=str(config_path))

        X_train = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [0.5, 1.5, 2.5]})
        X_test = pd.DataFrame({"feature1": [1.5], "feature2": [1.0]})

        X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)

        assert X_train_scaled.min().min() >= 0.0
        assert X_train_scaled.max().max() <= 1.0

    def test_save_as_csv(self, temp_config):
        """Test saving DataFrame to CSV."""
        from src.transform import WeatherDiseasePreprocessor

        config_path = test_data_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(temp_config, f)

        processor = WeatherDiseasePreprocessor(config_path=str(config_path))

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        csv_path = test_data_dir / "test_output.csv"

        processor.save_as_csv(df, csv_path)

        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 3

    def test_save_pickle(self, temp_config):
        """Test saving object to pickle file."""
        from src.transform import WeatherDiseasePreprocessor

        config_path = test_data_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(temp_config, f)

        processor = WeatherDiseasePreprocessor(config_path=str(config_path))

        obj = {"key": "value", "list": [1, 2, 3]}
        pickle_path = test_data_dir / "test.pkl"

        processor.save_pickle(obj, pickle_path)

        assert pickle_path.exists()
        with open(pickle_path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == obj


