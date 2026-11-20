# DVC (Data Version Control)

This directory contains DVC configuration for version-controlling data and model artifacts.

## Quick Start

```bash
# Track a data file
dvc add data/raw/Weather-related\ disease\ prediction.csv

# Reproduce the pipeline
make dvc
# or
dvc repro

# Push data to remote
dvc push

# Pull data from remote
dvc pull
```

## Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| clean | `dvc repro clean` | Clean raw data, add UUID, remove duplicates |
| validate | `dvc repro validate` | Enforce the data contract |
| transform | `dvc repro transform` | Split, encode, save raw features |
| train | `dvc repro train` | Train and register `@champion` |
| evaluate | `dvc repro evaluate` | Metrics, reference, and plots |
| ablation | `dvc repro ablation` | Weather vs symptoms vs full |
| calibration | `dvc repro calibration` | Brier/ECE + reliability diagram |
| slices | `dvc repro slices` | Age/gender error slices |
| cross_validate | `dvc repro cross_validate` | 5-fold CV intervals |
| explain | `dvc repro explain` | SHAP attribution |

## Remote Storage

Configured remote: `origin` (Dagshub)
Run `dvc remote add` to configure your own S3/GCS bucket.