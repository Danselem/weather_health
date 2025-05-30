"""
Script to split, transform, and save the weather disease dataset
using a class-based design.

This preprocessing pipeline:
- Loads a cleaned dataset from a Parquet file
- Removes unnecessary columns (e.g., 'uuid')
- Splits the data into training and testing sets
- Encodes categorical target labels
- Scales feature columns using MinMaxScaler
- Saves transformed datasets as CSV files
- Saves the scaler and encoder objects as .pkl for reuse
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Tuple
import pickle
from src import logger


class WeatherDiseasePreprocessor:
    """
    A class to preprocess the Weather Disease dataset.
    
    Attributes:
        input_path (Path): Path to the input cleaned dataset in Parquet format.
        output_dir (Path): Directory where all processed outputs are saved.
        test_size (float): Proportion of data to allocate to the test set.
        random_state (int): Seed for reproducibility in data splitting.
        scaler (MinMaxScaler): Scaler for feature normalization.
        label_encoder (LabelEncoder): Encoder for target label transformation.
    """

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        test_size: float = 0.2,
        random_state: int = 1024,
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """
        Load the cleaned dataset from a Parquet file.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        logger.info(f"Loading data from {self.input_path}")
        return pd.read_parquet(self.input_path)

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and testing sets.

        Args:
            data (pd.DataFrame): Full dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test sets.
        """
        logger.info("Splitting data into train and test sets")
        return train_test_split(data, test_size=self.test_size, random_state=self.random_state)

    def encode_labels(self, y: pd.Series) -> pd.Series:
        """
        Encode target labels as integers using LabelEncoder.

        Args:
            y (pd.Series): Target column.

        Returns:
            pd.Series: Encoded labels.
        """
        logger.info("Encoding target labels")
        return pd.Series(self.label_encoder.fit_transform(y), name=y.name)

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numeric features using MinMaxScaler.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and testing features.
        """
        logger.info("Scaling features using MinMaxScaler")
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)
        return X_train_scaled, X_test_scaled

    def save_as_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save a DataFrame as a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            filename (str): File name for CSV output.
        """
        path = self.output_dir / filename
        logger.info(f"Saving {filename} to {path}")
        df.to_csv(path, index=False)

    def save_pickle_object(self, obj, filename: str) -> None:
        """
        Save a Python object as a pickle file.

        Args:
            obj: Object to be serialized.
            filename (str): Pickle file name.
        """
        path = self.output_dir / filename
        logger.info(f"Saving {filename} to {path}")
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def run(self) -> None:
        """
        Run the full preprocessing pipeline:
        - Load data
        - Drop 'uuid' column if present
        - Split into train/test sets
        - Encode labels
        - Scale features
        - Save datasets and transformers
        """
        logger.info("Starting weather disease data preprocessing")

        # Load and clean data
        data = self.load_data()
        if 'uuid' in data.columns:
            logger.info("Dropping 'uuid' column")
            data.drop(columns=["uuid"], inplace=True)

        # Split data
        df_train, df_test = self.split_data(data)

        # Separate features and target
        X_train, y_train = df_train.drop(columns=["prognosis"]), df_train["prognosis"]
        X_test, y_test = df_test.drop(columns=["prognosis"]), df_test["prognosis"]

        # Encode target labels
        y_train_encoded = self.encode_labels(y_train)
        y_test_encoded = self.encode_labels(y_test)

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        # Save datasets as CSV
        self.save_as_csv(X_train_scaled, "x_train.csv")
        self.save_as_csv(y_train_encoded.to_frame(), "y_train.csv")
        self.save_as_csv(X_test_scaled, "x_test.csv")
        self.save_as_csv(y_test_encoded.to_frame(), "y_test.csv")

        # Save the scaler and label encoder
        self.save_pickle_object(self.scaler, "minmax_scaler.pkl")
        self.save_pickle_object(self.label_encoder, "label_encoder.pkl")

        logger.info("Data preprocessing and saving completed successfully.")


def main():
    params_path = Path("params.yaml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    
    input_path = params["data"]["interim_data_path"]
    output_dir = params["data"]["output_dir"]
    
    processor = WeatherDiseasePreprocessor(input_path=input_path, output_dir=output_dir)
    processor.run()


if __name__ == "__main__":
    main()
