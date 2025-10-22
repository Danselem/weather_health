## Weather–Health: Building a Production‑Ready Weather‑Disease Predictor

### Executive Summary
Weather influences the prevalence and severity of several diseases (e.g., flu, asthma, heat‑related conditions). This repository delivers a reproducible ML system that predicts disease categories from environmental and symptom data, with a focus on operational quality: experiment tracking, reliable training, containerized serving, and live monitoring.

### Objectives
- Deliver a robust end‑to‑end pipeline (ingest → clean → transform → train → register best model).
- Serve a FastAPI endpoint for low‑latency inference in Docker.
- Track experiments with MLflow (via DagsHub or AWS backend).
- Monitor model quality and data drift using Evidently + PostgreSQL + Grafana.

---

## Data and Features
- Source: Kaggle weather‑related disease dataset (see project `README.md` for link).
- Typical features: `Age`, `Gender`, `Temperature (C)`, `Humidity`, `Wind Speed (km/h)`, symptom indicators (e.g., `high_fever`, `back_pain`, `chills`, `joint_pain`).
- Target: multiclass disease label (label‑encoded for training and metrics).

Data access, paths, and column names are configured in `params.yaml`. Processed artifacts (scaler, label encoder, reference data) are stored under `data/processed/`.

---

## Pipeline Architecture
The pipeline is orchestrated with Prefect and exposed as simple `make` commands. Core components:

- Exploratory analysis: `src/visualise.py` exports plots and summary statistics for quick data understanding.
- Cleaning: `src/clean_data.py` handles missing values, types, and writes interim data.
- Transformation: `src/transform.py` produces training‑ready features and saves artifacts (MinMax scaler, label encoder).
- Training: `src/train.py` runs Hyperopt‑based tuning and model registration to MLflow; `src/utils/mlflow_manager.py` holds model registration utilities.
- Orchestration: `src/pipeline.py` wires the stages and can be triggered via `make pipeline`.

Artifacts are saved under `models/` and `data/processed/`. Configuration is centralized in `params.yaml` to keep runs reproducible.

---

## Experiment Tracking
The project logs runs, parameters, and metrics to MLflow:

- DagsHub option: store experiments remotely by setting your DagsHub credentials in `.env` and following `README.md` instructions.
- AWS option: Terraform modules under `infra/` provision EC2, RDS, S3, and Kinesis/Lambda for a cloud‑hosted backend.

Registered models can be pulled locally via `make fetch-best-model` once your experiments are complete.

---

## Modeling and Evaluation
- Models: Random Forest, Gradient Boosting, Logistic Regression, Histogram‑based GB, and LightGBM.
- Search: Hyperopt optimizes hyperparameters with a configurable loss; see `params.yaml` → `modeling`.
- Metrics: Accuracy, Precision, Recall, and F1; reports and plots are saved to `reports/`.
- Explainability: Classical feature importances are exported for supported models to aid interpretation.

---

## Serving (FastAPI + Docker)
The `app/main.py` FastAPI service loads the trained model, scaler, and label encoder, enforces feature order, and exposes a JSON API.

Start locally with Docker:

```bash
make build
make run
```

Send a prediction request (service listens on port 9696):

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d @sample_input.json \
  http://localhost:9696/predict
```

Alternatively, run an offline local check without the API:

```bash
make sample       # generate a valid input JSON
make serve_local  # run local inference on the sample
```

---

## Monitoring and Observability
The repository ships a minimal monitoring stack to simulate batch‑style observability and drift detection with Evidently, PostgreSQL, Adminer, and Grafana.

Start services and stream metrics:

```bash
make start-monitoring   # docker‑compose up (PostgreSQL, Adminer, Grafana)
make observe            # run Prefect flow to compute and write Evidently metrics
```

Dashboards:
- Adminer at `http://localhost:8080` (database inspection).
- Grafana at `http://localhost:3000` (preconfigured dashboard for drift/quality).

The monitoring flow (`src/evidently_metrics.py`) computes value drift on predictions, counts drifted columns, and tracks missing values. Metrics are pushed to PostgreSQL at a fixed cadence to emulate production batches.

---

## Reproducible Quickstart
```bash
# 1) Environment
make init
make install

# 2) (Optional) Start Prefect UI locally
make prefect

# 3) End‑to‑end pipeline
make pipeline

# 4) Fetch best model from MLflow registry (DagsHub/AWS configured)
make fetch-best-model

# 5) Build and run API
make build && make run

# 6) Request a prediction
curl -H "Content-Type: application/json" -d @sample_input.json http://localhost:9696/predict
```

---

## Users and Use Cases
- Public‑health teams: early warnings, triage planning, and communication.
- Data teams: reproducible training/evaluation, audit trails via MLflow.
- Platform teams: containerized serving + minimal Ops for monitoring.

---

## Limitations and Next Steps
- Dataset scope: kaggle‑sourced; field validation and external generalization are recommended before clinical use.
- Interpretability: SHAP/PDP are out‑of‑scope in this baseline; can be added for deeper insight.
- Automation: CI/CD and scheduled retraining can be integrated with Prefect deployments and cloud infra.

Planned improvements include automated retraining on drift alerts, richer model explanations, and multi‑region benchmarking.

---

## Links and Credits
- Repository: https://github.com/Danselem/weather-health
- Experiments: https://dagshub.com/Danselem/weather_health/experiments

Thanks to the open‑source community for Prefect, MLflow, Evidently, Grafana, and the dataset contributors.