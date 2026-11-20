"""
Main ML pipeline orchestration with Hydra configuration support.

Usage:
    python pipeline.py                  # Default: dev environment
    python pipeline.py env=prod          # Production environment
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.clean_data import DataCleaner
from src.config import load_config
from src.train import ModelTrainer
from src.transform import WeatherDiseasePreprocessor
from src.visualise import EDAReport

load_dotenv()


class Pipeline:
    """Main ML pipeline orchestrator using Hydra configuration."""

    def __init__(self, config_path: str | None = None, overrides: list[str] | None = None):
        self.overrides = overrides or []
        self.cfg: DictConfig | None = None
        self._load_config(config_path)

    def _get_config_dir(self) -> Path:
        return Path(__file__).parent.parent / "config"

    def _load_config(self, config_path: str | None) -> None:
        self.cfg = load_config(config_path or self._get_config_dir(), self.overrides)

    def run(self) -> None:
        """Execute the full ML pipeline."""
        data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        eda_cfg = self.cfg.eda

        print(f"Running pipeline in {self.cfg.env.name} environment")
        print(f"Model: {self.cfg.modeling.model_family}")

        print("\n=== Stage 1: EDA ===")
        data = pd.read_csv(Path(data_cfg["raw_data_path"]))
        eda = EDAReport(data, target_col=data_cfg["target_col"], output_prefix=eda_cfg.output_prefix)
        eda.generate_report()

        print("\n=== Stage 2: Data Cleaning ===")
        cleaner = DataCleaner(overrides=self.overrides)
        cleaner.run()

        print("\n=== Stage 3: Data Transformation ===")
        processor = WeatherDiseasePreprocessor(overrides=self.overrides)
        processor.preprocess_data()

        print("\n=== Stage 4: Model Training ===")
        trainer = ModelTrainer(overrides=self.overrides)
        trainer.run()

        print("\n=== Stage 5: Evaluation ===")
        from src.evaluate import main as evaluate_main

        evaluate_main()

        print("\n=== Pipeline Complete ===")


def main(config_path: str | None = None, overrides: list[str] | None = None) -> None:
    pipeline = Pipeline(config_path=config_path, overrides=overrides)
    pipeline.run()


if __name__ == "__main__":
    import sys

    main(overrides=[arg for arg in sys.argv[1:] if "=" in arg])
