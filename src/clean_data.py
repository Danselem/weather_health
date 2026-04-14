"""DataCleaner class for preprocessing a weather-related disease dataset with Hydra config."""

import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from prefect import flow, task

from src import logger

load_dotenv()


class DataCleaner:
    """
    A class to clean and preprocess a dataset for weather-related disease prediction.

    This includes:
    - Loading data from CSV
    - Adding a UUID column
    - Dropping specified columns
    - Removing duplicate rows
    - Saving the cleaned data to a Parquet file

    Attributes:
        input_path (Path): Path to the raw input CSV file.
        output_path (Path): Path to save the cleaned output Parquet file.
        data (pd.DataFrame): The working DataFrame.
    """

    def __init__(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        config_path: str | None = None,
    ) -> None:
        """
        Initialize the DataCleaner with Hydra configuration.

        Args:
            input_path: File path to the input CSV (optional - uses config if not set).
            output_path: File path to save the cleaned Parquet data (optional - uses config).
            config_path: Path to Hydra config directory (optional).
        """
        self._load_config(config_path)

        data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)

        self.input_path: Path = Path(input_path) if input_path else Path(data_cfg["raw_data_path"])
        self.output_path: Path = Path(output_path) if output_path else Path(data_cfg["interim_data_path"])
        self.data: pd.DataFrame | None = None

        logger.info(f"DataCleaner initialized for environment: {self.cfg.env.env.name}")

    def _load_config(self, config_path: str | None) -> None:
        config_dir = Path(config_path) if config_path else self._get_config_dir()
        if not config_dir.is_absolute():
            config_dir = Path.cwd() / config_dir

        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_dir.resolve()),
        ):
            self.cfg = compose(config_name="config")
            OmegaConf.resolve(self.cfg)

    def _get_config_dir(self) -> Path:
        return Path(__file__).parent.parent / "config"

    @task(name="load_data", retries=3, retry_delay_seconds=10, log_prints=True)
    def load_data(self) -> None:
        """Load the dataset from the CSV file into a pandas DataFrame."""
        self.data = pd.read_csv(self.input_path)
        logger.info(f"Loaded data from {self.input_path} with shape {self.data.shape}")

    @task(name="add_uuid_column", retries=3, retry_delay_seconds=10, log_prints=True)
    def add_uuid_column(
        self,
        exclude_cols: list[str] | None = None,
        namespace: uuid.UUID = uuid.NAMESPACE_DNS,
    ) -> None:
        """
        Add a UUID column based on the hash of each row, excluding specified columns.

        Args:
            exclude_cols (Optional[List[str]]): Columns to exclude from UUID generation.
            namespace (uuid.UUID): UUID namespace for deterministic UUID5 generation.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        if exclude_cols is None:
            cols_to_use = self.data.columns
        else:
            cols_to_use = [col for col in self.data.columns if col not in exclude_cols]

        def row_to_string(row: pd.Series) -> str:
            return "_".join(str(value) for value in row)

        uuids = self.data[cols_to_use].apply(
            lambda row: uuid.uuid5(namespace, row_to_string(row)), axis=1
        )
        self.data.insert(0, "uuid", uuids.astype(str))

        logger.info("UUID column added")
        logger.info("UUID column shape: %s", self.data.shape)

    @task(name="clean_data", retries=3, retry_delay_seconds=10, log_prints=True)
    def clean_data(self) -> None:
        """
        Clean the dataset by removing specified columns and duplicate rows.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        logger.info("Original data shape: %s", self.data.shape)

        cols_to_drop: list[str] = [
            "shivering",
            "asthma_history",
            "high_cholesterol",
            "diabetes",
            "obesity",
        ]
        self.data.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        logger.info("Dropped %d columns: %s", len(cols_to_drop), cols_to_drop)
        logger.info("Data shape after dropping columns: %s", self.data.shape)

        duplicated = self.data.duplicated().sum()
        logger.info("Duplicated rows: %d", duplicated)

        self.data.drop_duplicates(inplace=True)
        logger.info("Data shape after dropping duplicates: %s", self.data.shape)

    def save_data(self) -> None:
        """Save the cleaned DataFrame to a Parquet file using pyarrow."""
        if self.data is None:
            raise ValueError("No data to save. Make sure cleaning has been run.")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_parquet(self.output_path, index=False, engine="pyarrow")
        logger.info(f"Data saved to {self.output_path}")

    @flow(name="clean_data", retries=3, retry_delay_seconds=10, log_prints=True)
    def run(self) -> None:
        """
        Run the full data cleaning pipeline:
        - Load data
        - Add UUID column
        - Clean data
        - Save cleaned data
        """
        logger.info(f"Starting data cleaning pipeline (env: {self.cfg.env.env.name})")
        self.load_data()
        self.add_uuid_column(exclude_cols=["prognosis"])
        self.clean_data()
        self.save_data()


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()
