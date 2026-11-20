# ADR 0003: Preprocessing pipeline and `@champion`

## Status
Accepted

## Context
Scaling was applied separately in training, serving, and batch scripts. The
registry selected a "best" run with an incorrect metric sort and no stable
deployment pointer.

## Decision
Log one scikit-learn pipeline (scaler + classifier). Select the lowest validation
loss and promote it to MLflow alias `@champion`. The test set is used once.

## Consequences
- Training-serving skew is removed; serving accepts raw features.
- `models:/...@champion` is the only deployment contract.
- Rollbacks and promotions are explicit alias changes.
