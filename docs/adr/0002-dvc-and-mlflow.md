# ADR 0002: DVC for data, MLflow for model lineage

## Status
Accepted

## Context
Data, models, and reports were committed directly to Git, making DVC decorative
and creating large binary history.

## Decision
DVC owns data, model, reference, metrics, and plot outputs. MLflow owns
experiments, signatures, and the model registry. Git owns code, config, DVC
pointers, and `dvc.lock`.

## Consequences
- A fresh clone requires `dvc pull`.
- CI needs DagsHub credentials.
- Model promotion is traceable to both a DVC hash and an MLflow run/version.
