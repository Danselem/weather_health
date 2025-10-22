from pathlib import Path

import pandas as pd
import yaml

from src.clean_data import DataCleaner
from src.train import ModelTrainer
from src.transform import WeatherDiseasePreprocessor
from src.visualise import EDAReport

# Load configuration
params_file = Path("params.yaml")
config = yaml.safe_load(open(params_file, encoding="utf-8"))

data_path = config["data"]["raw_data_path"]
target_col = config["data"]["target_col"]
output_prefix = config["eda"]["output_prefix"]

data_path = config["data"]["raw_data_path"]
output_path = config["data"]["interim_data_path"]

input_path = config["data"]["interim_data_path"]
output_dir = config["data"]["output_dir"]

data = pd.read_csv(data_path)
eda = EDAReport(data, target_col=target_col, output_prefix=output_prefix)
eda.generate_report()

cleaner = DataCleaner(input_path=data_path, output_path=output_path)
cleaner.run()

processor = WeatherDiseasePreprocessor(config_path=params_file)
processor.preprocess_data()


trainer = ModelTrainer(str(params_file))
trainer.run()
