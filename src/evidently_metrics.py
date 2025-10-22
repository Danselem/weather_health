import datetime
import logging
import pickle
import random
import time

import pandas as pd
import psycopg
import pytz
import yaml
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, MissingValueCount, ValueDrift
from prefect import flow, task

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

params = yaml.safe_load(open("params.yaml"))


reference_data = pd.read_csv("data/processed/reference.csv")

with open(params["artifacts"]["model_path"], "rb") as f_in:
    model = pickle.load(f_in)

scaler = pickle.load(open(params["artifacts"]["scaler_path"], "rb"))
encoder = pickle.load(open(params["artifacts"]["label_encoder_path"], "rb"))


raw_data = pd.read_csv(params["data"]["x_train_path"])
raw_data = pd.DataFrame(scaler.transform(raw_data), columns=raw_data.columns)

begin = datetime.datetime(2022, 2, 1, 0, 0)
num_features = ["Age", "Temperature (C)", "Humidity", "Wind Speed (km/h)"]
cat_features = [col for col in raw_data.columns if col not in num_features]

data_definition = DataDefinition(
    numerical_columns=num_features + ["prediction"],
    categorical_columns=cat_features,
)

report = Report(
    metrics=[
        ValueDrift(column="prediction"),
        DriftedColumnsCount(),
        MissingValueCount(column="prediction"),
    ]
)


CONNECTION_STRING = "host=localhost port=5432 user=postgres password=example"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname=test"


@task
def prep_db():
    with psycopg.connect(CONNECTION_STRING, autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(CONNECTION_STRING_DB) as conn:
            conn.execute(create_table_statement)


# @task
# def calculate_metrics_postgresql(i):
# 	current_data = raw_data[(raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
# 		(raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]

# 	#current_data.fillna(0, inplace=True)
# 	current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))

# 	current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)
# 	reference_dataset = Dataset.from_pandas(reference_data, data_definition=data_definition)

# 	run = report.run(reference_data=reference_dataset, current_data=current_dataset)

# 	result = run.dict()

# 	prediction_drift = result['metrics'][0]['value']
# 	num_drifted_columns = result['metrics'][1]['value']['count']
# 	share_missing_values = result['metrics'][2]['value']['share']
# 	with psycopg.connect(CONNECTION_STRING_DB, autocommit=True) as conn:
# 		with conn.cursor() as curr:
# 			curr.execute(
# 				"insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
# 				(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values)
# 			)


@task
def calculate_metrics_postgresql(i):
    """
    Runs Evidently on one batch and logs metrics to PostgreSQL.
    ─────────────────────────────────────────────────────────────
    Keeps the same signature and the BEGIN + i days timestamp.
    Otherwise, it falls back to the original date‑slice behaviour.
    """

    current_data = raw_data.copy()  # use all rows; no datetime col

    # ── 2. add model predictions ───────────────────────────────
    features = current_data[num_features + cat_features].fillna(0)
    current_data = current_data.copy()
    current_data["prediction"] = model.predict(features)

    # ── 3. Evidently datasets & run ────────────────────────────
    current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)
    reference_dataset = Dataset.from_pandas(
        reference_data, data_definition=data_definition
    )

    result = report.run(
        reference_data=reference_dataset, current_data=current_dataset
    ).dict()

    prediction_drift = result["metrics"][0]["value"]
    num_drifted_columns = result["metrics"][1]["value"]["count"]
    share_missing_values = result["metrics"][2]["value"]["share"]

    # ── 4. write metrics to PostgreSQL ─────────────────────────
    ts = datetime.datetime.now(
        pytz.timezone("Africa/Johannesburg")
    )  # retain original timestamp logic

    with psycopg.connect(CONNECTION_STRING_DB, autocommit=True) as conn:
        with conn.cursor() as curr:
            curr.execute(
                """
                INSERT INTO dummy_metrics
                (timestamp, prediction_drift, num_drifted_columns, share_missing_values)
                VALUES (%s, %s, %s, %s)
                """,
                (ts, prediction_drift, num_drifted_columns, share_missing_values),
            )


@flow
def batch_monitoring():
    prep_db()
    last_send = datetime.datetime.now(
        pytz.timezone("Africa/Johannesburg")
    ) - datetime.timedelta(seconds=15)
    for i in range(0, 200):
        calculate_metrics_postgresql(i)

        new_send = datetime.datetime.now(pytz.timezone("Africa/Johannesburg"))
        seconds_elapsed = (new_send - last_send).total_seconds()
        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)
        while last_send < new_send:
            last_send = last_send + datetime.timedelta(seconds=15)
        logging.info("data sent")


if __name__ == "__main__":
    batch_monitoring()
