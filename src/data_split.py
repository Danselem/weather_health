"""Script to split the Indicators of Heart Disease dataset into training and
testing sets using a class-based design."""

from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src import logger


class DataSplitter:
    """
    A class to handle loading, splitting, and saving a dataset
    for training and testing purposes.

    Attributes:
        raw_data_path (Path): Path to the input Parquet file.
        train_output_path (Path): Path to save the training dataset.
        test_output_path (Path): Path to save the testing dataset.
        test_size (float): Fraction of data to reserve for testing.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        raw_data_path: Union[
            str, Path
        ] = "project1/data/interim/weather_disease.parquet",
        train_output_path: Union[
            str, Path
        ] = "project1/data/interim/weather_disease_train.parquet",
        test_output_path: Union[
            str, Path
        ] = "project1/data/interim/weather_disease_test.parquet",
        test_size: float = 0.2,
        random_state: int = 1024,
    ) -> None:
        """
        Initialize the DataSplitter with file paths and split parameters.

        Args:
            raw_data_path (Union[str, Path]): Path to the raw dataset.
            train_output_path (Union[str, Path]): Path to save the training set.
            test_output_path (Union[str, Path]): Path to save the testing set.
            test_size (float): Fraction of data to use for testing.
            random_state (int): Random seed for reproducibility.
        """
        self.raw_data_path = Path(raw_data_path)
        self.train_output_path = Path(train_output_path)
        self.test_output_path = Path(test_output_path)
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from a Parquet file.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        logger.info(f"Loading dataset from {self.raw_data_path}")
        data = pd.read_parquet(self.raw_data_path)
        logger.info(f"Dataset loaded successfully with shape {data.shape}")
        return data

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and testing sets.

        Args:
            data (pd.DataFrame): The dataset to split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets.
        """
        logger.info("Splitting the data into training and testing sets")
        df_train, df_test = train_test_split(
            data, test_size=self.test_size, random_state=self.random_state
        )
        logger.info(f"Training set shape: {df_train.shape}")
        logger.info(f"Testing set shape: {df_test.shape}")
        return df_train, df_test

    def save_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
        """
        Save the training and testing datasets to Parquet files.

        Args:
            df_train (pd.DataFrame): Training dataset.
            df_test (pd.DataFrame): Testing dataset.
        """
        df_train.to_parquet(self.train_output_path, index=False)
        df_test.to_parquet(self.test_output_path, index=False)
        logger.info(f"Training data saved to {self.train_output_path}")
        logger.info(f"Testing data saved to {self.test_output_path}")

    def run(self) -> None:
        """
        Run the full data splitting pipeline.
        """
        logger.info("Starting data split process")
        data = self.load_data()
        df_train, df_test = self.split_data(data)
        self.save_data(df_train, df_test)
        logger.info("Data split and save process completed successfully")


def main():
    splitter = DataSplitter()
    splitter.run()


if __name__ == "__main__":
    main()
