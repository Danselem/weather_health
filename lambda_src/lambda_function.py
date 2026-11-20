import json
import os
import pickle
from pathlib import Path

import pandas as pd

BASE = Path(os.getenv("MODEL_DIR", "."))
MODEL = pickle.load(open(BASE / "weather_disease_model.pkl", "rb"))
ENCODER = pickle.load(open(BASE / "label_encoder.pkl", "rb"))
FEATURE_ORDER = json.load(open(BASE / "feature_order.json", encoding="utf-8"))


def lambda_handler(event, context):
    outputs = []
    for record in event["Records"]:
        payload = json.loads(record["kinesis"]["data"])
        df = pd.DataFrame([[payload[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
        pred_code = MODEL.predict(df)[0]
        outputs.append(
            {
                "uuid": payload.get("uuid"),
                "prediction": ENCODER.inverse_transform([pred_code])[0],
            }
        )
    return {"predictions": outputs}
