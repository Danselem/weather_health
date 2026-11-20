import base64
import json
import pickle

import pandas as pd

MODEL = pickle.load(open("weather_disease_model.pkl", "rb"))
ENCODER = pickle.load(open("label_encoder.pkl", "rb"))
FEATURE_ORDER = json.load(open("feature_order.json", encoding="utf-8"))


def lambda_handler(event, context):
    outputs = []
    for record in event["Records"]:
        payload = json.loads(base64.b64decode(record["kinesis"]["data"]))
        df = pd.DataFrame([[payload[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
        pred_code = MODEL.predict(df)[0]
        outputs.append(
            {
                "uuid": payload.get("uuid"),
                "prediction": ENCODER.inverse_transform([pred_code])[0],
            }
        )
    return {"predictions": outputs}
