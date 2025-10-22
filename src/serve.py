import json
from pathlib import Path

import requests


def load_sample_and_predict():
    """Load sample JSON data and send it to the model API for prediction."""
    # Load the sample JSON file
    sample_path = Path("sample_input.json")
    with open(sample_path, "r", encoding="utf-8") as file:
        sample_data = json.load(file)
        # print(sample_data)

    # API URL (make sure it's correct, assuming it's running locally)
    api_url = "http://localhost:8080/predict"

    # Make a POST request to the API with the sample data
    response = requests.post(api_url, json=sample_data)

    # Check if the request was successful
    if response.status_code == 200:
        prediction = response.json()
        print(f"Prediction result: {prediction}")
    else:
        print(f"Failed to get prediction. Status code: {response.status_code}")
        print(f"Error response: {response.text}")


if __name__ == "__main__":
    load_sample_and_predict()
