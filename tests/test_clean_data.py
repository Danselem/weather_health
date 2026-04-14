"""Unit tests for DataCleaner class."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


class TestDataCleaner:
    """Tests for the DataCleaner class."""

    @pytest.fixture
    def temp_csv(self, test_data_dir):
        """Create a temporary CSV file for testing."""
        csv_path = test_data_dir / "test_input.csv"
        df = pd.DataFrame(
            {
                "Age": [25, 30, 35, 25],
                "Gender": ["Male", "Female", "Male", "Male"],
                "Temperature (C)": [37.0, 38.5, 36.5, 37.0],
                "prognosis": ["Flu", "Malaria", "Dengue", "Flu"],
                "shivering": [1, 0, 1, 1],
                "asthma_history": [0, 1, 0, 0],
                "diabetes": [0, 0, 1, 0],
            }
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def cleaner(self, temp_csv, test_data_dir):
        """Create a DataCleaner instance with temporary paths."""
        from src.clean_data import DataCleaner

        output_path = test_data_dir / "output.parquet"
        return DataCleaner(input_path=temp_csv, output_path=output_path)

    def test_init(self, temp_csv, test_data_dir):
        """Test DataCleaner initialization."""
        from src.clean_data import DataCleaner

        output_path = test_data_dir / "output.parquet"
        cleaner = DataCleaner(input_path=temp_csv, output_path=output_path)

        assert cleaner.input_path == Path(temp_csv)
        assert cleaner.output_path == Path(output_path)
        assert cleaner.data is None

    def test_load_data(self, cleaner):
        """Test loading data from CSV."""
        cleaner.load_data()

        assert cleaner.data is not None
        assert isinstance(cleaner.data, pd.DataFrame)
        assert len(cleaner.data) == 4
        assert "Age" in cleaner.data.columns

    def test_load_data_file_not_found(self, test_data_dir):
        """Test loading non-existent file raises error."""
        from src.clean_data import DataCleaner

        cleaner = DataCleaner(
            input_path="nonexistent.csv",
            output_path=test_data_dir / "output.parquet",
        )

        with pytest.raises(FileNotFoundError):
            cleaner.load_data()

    def test_add_uuid_column(self, cleaner):
        """Test adding UUID column to data."""
        cleaner.load_data()
        cleaner.add_uuid_column(exclude_cols=["prognosis"])

        assert "uuid" in cleaner.data.columns
        assert len(cleaner.data) == 4

    def test_add_uuid_column_without_loaded_data(self, cleaner):
        """Test adding UUID without loading data raises error."""
        with pytest.raises(ValueError, match="Data is not loaded"):
            cleaner.add_uuid_column()

    def test_clean_data(self, cleaner):
        """Test cleaning data by dropping columns and duplicates."""
        cleaner.load_data()
        cleaner.clean_data()

        assert "shivering" not in cleaner.data.columns
        assert "asthma_history" not in cleaner.data.columns
        assert "diabetes" not in cleaner.data.columns
        assert len(cleaner.data) == 3

    def test_clean_data_without_loaded_data(self, cleaner):
        """Test cleaning without loading data raises error."""
        with pytest.raises(ValueError, match="Data is not loaded"):
            cleaner.clean_data()

    def test_save_data(self, cleaner, test_data_dir):
        """Test saving cleaned data to parquet."""
        cleaner.load_data()
        cleaner.clean_data()
        cleaner.save_data()

        assert cleaner.output_path.exists()
        loaded = pd.read_parquet(cleaner.output_path)
        assert len(loaded) == 3

    def test_save_data_without_cleaning(self, cleaner):
        """Test saving without cleaning raises error."""
        cleaner.load_data()

        with pytest.raises(ValueError, match="No data to save"):
            cleaner.save_data()

    @patch("src.clean_data.logger")
    def test_run_full_pipeline(self, mock_logger, cleaner):
        """Test full pipeline execution."""
        cleaner.run()

        assert cleaner.output_path.exists()
        loaded = pd.read_parquet(cleaner.output_path)
        assert "uuid" in loaded.columns
        assert "prognosis" in loaded.columns
