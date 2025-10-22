import json
import pickle

import pandas as pd

# Load model and preprocessors from local files in the root of the ZIP package
MODEL = pickle.load(open("weather_disease_model.pkl", "rb"))
SCALER = pickle.load(open("minmax_scaler.pkl", "rb"))
ENCODER = pickle.load(open("label_encoder.pkl", "rb"))
FEATURE_ORDER = list(SCALER.feature_names_in_)


def lambda_handler(event, context):
    outputs = []
    for record in event["Records"]:
        # Kinesis base64-decodes 'data' automatically, but if not, decode here
        payload = json.loads(record["kinesis"]["data"])
        # Construct DataFrame in correct feature order
        df = pd.DataFrame(
            [[payload[feature] for feature in FEATURE_ORDER]], columns=FEATURE_ORDER
        )
        scaled = pd.DataFrame(SCALER.transform(df), columns=FEATURE_ORDER)
        pred_code = MODEL.predict(scaled)[0]
        prediction = ENCODER.inverse_transform([pred_code])[0]

        outputs.append({"uuid": payload.get("uuid"), "prediction": prediction})

    return {"predictions": outputs}
