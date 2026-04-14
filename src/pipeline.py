"""
Main ML pipeline orchestration with Hydra configuration support.

Usage:
    python pipeline.py                  # Default: dev environment
    python pipeline.py env=prod          # Production environment
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.clean_data import DataCleaner
from src.train import ModelTrainer
from src.transform import WeatherDiseasePreprocessor
from src.visualise import EDAReport

load_dotenv()


class Pipeline:
    """Main ML pipeline orchestrator using Hydra configuration."""

    def __init__(self, config_path: Optional[str] = None):
        self.cfg: Optional[DictConfig] = None
        self._load_config(config_path)

    def _get_config_dir(self) -> Path:
        return Path(__file__).parent.parent / "config"

    def _load_config(self, config_path: Optional[str]) -> None:
        config_dir = Path(config_path) if config_path else self._get_config_dir()
        if not config_dir.is_absolute():
            config_dir = Path.cwd() / config_dir

        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_dir.resolve()),
        ):
            self.cfg = compose(config_name="config")
            OmegaConf.resolve(self.cfg)

    def run(self) -> None:
        """Execute the full ML pipeline."""
        data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        eda_cfg = self.cfg.eda

        print(f"Running pipeline in {self.cfg.env.env.name} environment")
        print(f"Model: {self.cfg.modeling.model_family}")

        print("\n=== Stage 1: EDA ===")
        data = pd.read_csv(Path(data_cfg["raw_data_path"]))
        eda = EDAReport(data, target_col=data_cfg["target_col"], output_prefix=eda_cfg.output_prefix)
        eda.generate_report()

        print("\n=== Stage 2: Data Cleaning ===")
        cleaner = DataCleaner()
        cleaner.run()

        print("\n=== Stage 3: Data Transformation ===")
        processor = WeatherDiseasePreprocessor()
        processor.preprocess_data()

        print("\n=== Stage 4: Model Training ===")
        trainer = ModelTrainer()
        trainer.run()

        print("\n=== Pipeline Complete ===")


def main(config_path: Optional[str] = None) -> None:
    pipeline = Pipeline(config_path=config_path)
    pipeline.run()


if __name__ == "__main__":
    main()