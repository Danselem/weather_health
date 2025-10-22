import json
import pickle

import pandas as pd

MODEL = pickle.load(open("/opt/model/weather_disease_model.pkl", "rb"))
SCALER = pickle.load(open("/opt/model/minmax_scaler.pkl", "rb"))
ENCODER = pickle.load(open("/opt/model/label_encoder.pkl", "rb"))
FEATURE_ORDER = list(SCALER.feature_names_in_)


def lambda_handler(event, context):
    # Kinesis delivers a batch; loop through records
    outputs = []
    for rec in event["Records"]:
        payload = json.loads(rec["kinesis"]["data"])  # base64 auto‑decoded
        df = pd.DataFrame([[payload[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
        scaled = pd.DataFrame(SCALER.transform(df), columns=FEATURE_ORDER)
        pred_code = MODEL.predict(scaled)[0]
        outputs.append(
            {
                "uuid": payload.get("uuid"),
                "prediction": ENCODER.inverse_transform([pred_code])[0],
            }
        )
    # push somewhere or just return
    return {"predictions": outputs}
