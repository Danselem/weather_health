# preprocess_weather_disease.py
"""
Data preprocessing module with Hydra configuration support.

Usage:
    python transform.py                      # Default config
    python transform.py env=prod             # Production config
"""

import pickle
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src import logger

load_dotenv()


class WeatherDiseasePreprocessor:
    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or self._get_config_dir()
        self.cfg: DictConfig | None = None

        self._load_config()
        self._init_processors()

    def _get_config_dir(self) -> Path:
        return Path(__file__).parent.parent / "config"

    def _load_config(self) -> None:
        config_dir = Path(self.config_path)
        if not config_dir.is_absolute():
            config_dir = Path.cwd() / config_dir

        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_dir.resolve()),
        ):
            self.cfg = compose(config_name="config")
            OmegaConf.resolve(self.cfg)

        self.data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.artifacts_cfg = OmegaConf.to_container(self.cfg.artifacts, resolve=True)
        self.data_split_cfg = OmegaConf.to_container(self.cfg.data_split, resolve=True)
        self.preprocessing_cfg = OmegaConf.to_container(self.cfg.preprocessing, resolve=True)

    def _init_processors(self) -> None:
        scaler_type = self.preprocessing_cfg.get("scaler_type", "minmax")
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

        self.label_encoder = LabelEncoder()

        data_paths = self.data_paths
        self.input_path = Path(data_paths["interim_data_path"])
        self.x_train_path = Path(data_paths["x_train_path"])
        self.y_train_path = Path(data_paths["y_train_path"])
        self.x_test_path = Path(data_paths["x_test_path"])
        self.y_test_path = Path(data_paths["y_test_path"])
        self.scaler_path = Path(self.artifacts_cfg["scaler_path"])
        self.encoder_path = Path(self.artifacts_cfg["label_encoder_path"])

        for path in [
            self.x_train_path,
            self.y_train_path,
            self.x_test_path,
            self.y_test_path,
            self.scaler_path,
            self.encoder_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def data_paths(self) -> dict:
        return self.data_cfg

    @task(name="load_data", retries=3, retry_delay_seconds=10, log_prints=True)
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_path}")
        return pd.read_parquet(self.input_path)

    @task(name="split_data", retries=3, retry_delay_seconds=10, log_prints=True)
    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        test_size = self.data_split_cfg.get("test_size", 0.2)
        random_state = self.data_split_cfg.get("random_state", 1024)
        stratify = self.data_split_cfg.get("stratify", True)

        logger.info(f"Splitting data (test_size={test_size}, random_state={random_state})")
        return train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=data["prognosis"] if stratify else None,
        )

    @task(name="encode_labels", retries=3, retry_delay_seconds=10, log_prints=True)
    def encode_labels(self, y: pd.Series) -> pd.Series:
        logger.info("Encoding target labels")
        return pd.Series(self.label_encoder.fit_transform(y), name=y.name)

    @task(name="scale_features", retries=3, retry_delay_seconds=10, log_prints=True)
    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Scaling features using MinMaxScaler")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=X_test.columns
        )
        return X_train_scaled, X_test_scaled

    @task(name="save_as_csv", retries=3, retry_delay_seconds=10, log_prints=True)
    def save_as_csv(self, df: pd.DataFrame, path: Path) -> None:
        logger.info(f"Saving CSV to {path}")
        df.to_csv(path, index=False)

    @task(name="save_pickle", retries=3, retry_delay_seconds=10, log_prints=True)
    def save_pickle(self, obj: object, path: Path) -> None:
        logger.info(f"Saving pickle to {path}")
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @task(name="preprocess_data", retries=3, retry_delay_seconds=10, log_prints=True)
    def preprocess_data(self) -> None:
        logger.info(f"Starting preprocessing pipeline (env: {self.cfg.env.env.name})")

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

        self.save_as_csv(X_train_scaled, self.x_train_path)
        self.save_as_csv(y_train_enc.to_frame(), self.y_train_path)
        self.save_as_csv(X_test_scaled, self.x_test_path)
        self.save_as_csv(y_test_enc.to_frame(), self.y_test_path)
        self.save_pickle(self.scaler, self.scaler_path)
        self.save_pickle(self.label_encoder, self.encoder_path)

        logger.info("Preprocessing completed and all files saved.")


@flow(name="preprocess_data", retries=3, retry_delay_seconds=10, log_prints=True)
def main(config_path: str | None = None) -> None:
    processor = WeatherDiseasePreprocessor(config_path=config_path)
    processor.preprocess_data()


if __name__ == "__main__":
    main()
