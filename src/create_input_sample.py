import json
import pickle

import pandas as pd
import yaml

DROP_COLUMNS = ["shivering", "asthma_history", "high_cholesterol", "diabetes", "obesity"]


def main():
    """Sample one raw training row and save raw + transformed JSON payloads."""
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    raw = pd.read_csv(params["data"]["raw_data_path"])
    target = params["data"]["target_col"]

    features = raw.drop(columns=[target] + DROP_COLUMNS, errors="ignore")
    sample = features.sample(1, random_state=1024)

    with open(params["artifacts"]["scaler_path"], "rb") as file:
        scaler = pickle.load(file)

    scaled = pd.DataFrame(
        scaler.transform(sample), columns=scaler.feature_names_in_
    )

    with open("sample_input.json", "w", encoding="utf-8") as file:
        json.dump(sample.iloc[0].to_dict(), file)

    with open("sample_input_transformed.json", "w", encoding="utf-8") as file:
        json.dump(scaled.iloc[0].to_dict(), file)


if __name__ == "__main__":
    main()
