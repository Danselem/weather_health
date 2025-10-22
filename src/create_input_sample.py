import json
import pickle
from pathlib import Path

import pandas as pd


def main():
    """Sample one training row, scale it, and save both raw and scaled versions."""
    data_path = Path("data/processed/x_train.csv")
    df_train = pd.read_csv(data_path)
    sample = df_train.sample(1)  # DataFrame, 1 row

    # ── Load fitted MinMaxScaler ───────────────────────────────────────────────
    with open(Path("data/processed/minmax_scaler.pkl"), "rb") as f_in:
        minmax_scaler = pickle.load(f_in)

    # ── Scale using a DF to keep feature names and avoid the warning ───────────
    sample_scaled_df = pd.DataFrame(
        minmax_scaler.transform(sample),  # 2‑D array
        columns=minmax_scaler.feature_names_in_,
    )

    # ── Build dictionaries -----------------------------------------------------
    raw_dict = sample.iloc[0].to_dict()
    transformed_dict = sample_scaled_df.iloc[0].to_dict()

    # ── Save JSON files --------------------------------------------------------
    with open("sample_input.json", "w", encoding="utf-8") as f:
        json.dump(raw_dict, f)

    with open("sample_input_transformed.json", "w", encoding="utf-8") as f:
        json.dump(transformed_dict, f)


if __name__ == "__main__":
    main()
