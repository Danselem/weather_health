# preprocess_weather_disease.py

import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src import logger


class WeatherDiseasePreprocessor:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.params = yaml.safe_load(f)

        self.input_path = Path(self.params["data"]["interim_data_path"])
        self.x_train_path = Path(self.params["data"]["x_train_path"])
        self.y_train_path = Path(self.params["data"]["y_train_path"])
        self.x_test_path = Path(self.params["data"]["x_test_path"])
        self.y_test_path = Path(self.params["data"]["y_test_path"])
        self.scaler_path = Path(self.params["artifacts"]["scaler_path"])
        self.encoder_path = Path(self.params["artifacts"]["label_encoder_path"])

        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

        # Ensure directories exist
        for path in [
            self.x_train_path,
            self.y_train_path,
            self.x_test_path,
            self.y_test_path,
            self.scaler_path,
            self.encoder_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)

    @task(name="load_data", retries=3, retry_delay_seconds=10, log_prints=True)
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_path}")
        return pd.read_parquet(self.input_path)

    @task(name="split_data", retries=3, retry_delay_seconds=10, log_prints=True)
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Splitting data into train and test sets")
        return train_test_split(data, test_size=0.2, random_state=1024)

    @task(name="encode_labels", retries=3, retry_delay_seconds=10, log_prints=True)
    def encode_labels(self, y: pd.Series) -> pd.Series:
        logger.info("Encoding target labels")
        return pd.Series(self.label_encoder.fit_transform(y), name=y.name)

    @task(name="scale_features", retries=3, retry_delay_seconds=10, log_prints=True)
    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Scaling features using MinMaxScaler")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=X_test.columns
        )
        return X_train_scaled, X_test_scaled

    @task(name="save_as_csv", retries=3, retry_delay_seconds=10, log_prints=True)
    def save_as_csv(self, df: pd.DataFrame, path: Path):
        logger.info(f"Saving CSV to {path}")
        df.to_csv(path, index=False)

    @task(name="save_pickle", retries=3, retry_delay_seconds=10, log_prints=True)
    def save_pickle(self, obj, path: Path):
        logger.info(f"Saving pickle to {path}")
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @task(name="preprocess_data", retries=3, retry_delay_seconds=10, log_prints=True)
    def preprocess_data(self):
        logger.info("Starting preprocessing pipeline")

        data = self.load_data()
        if "uuid" in data.columns:
            logger.info("Dropping 'uuid' column")
            data = data.drop(columns=["uuid"])

        train_df, test_df = self.split_data(data)

        X_train, y_train = train_df.drop(columns=["prognosis"]), train_df["prognosis"]
        X_test, y_test = test_df.drop(columns=["prognosis"]), test_df["prognosis"]

        y_train_enc = self.encode_labels(y_train)
        y_test_enc = self.encode_labels(y_test)

        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        # Save all outputs using configured paths
        self.save_as_csv(X_train_scaled, self.x_train_path)
        self.save_as_csv(y_train_enc.to_frame(), self.y_train_path)
        self.save_as_csv(X_test_scaled, self.x_test_path)
        self.save_as_csv(y_test_enc.to_frame(), self.y_test_path)
        self.save_pickle(self.scaler, self.scaler_path)
        self.save_pickle(self.label_encoder, self.encoder_path)

        logger.info("Preprocessing completed and all files saved.")


@flow(name="preprocess_data", retries=3, retry_delay_seconds=10, log_prints=True)
def main():
    processor = WeatherDiseasePreprocessor(config_path="params.yaml")
    processor.preprocess_data()


if __name__ == "__main__":
    main()
