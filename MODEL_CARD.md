# Model Card: Weather-Health Disease Classifier

## Model

- **Task:** 11-class disease classification from weather and symptom features.
- **Artifact:** scikit-learn `Pipeline(MinMaxScaler -> LGBMClassifier)`.
- **Registry:** MLflow model `lightgbm_best_model`, promoted via `@champion`.
- **Data version:** tracked by DVC (`dvc.lock`).
- **Feature order:** `data/processed/feature_order.json`.

## Intended use

Portfolio demonstration of an end-to-end MLOps system: data versioning,
experiment tracking, validation, evaluation, serving, and monitoring. It is
**not** a clinical decision-support tool and must not be used for diagnosis.

## Data

- Source: public "Weather-related disease prediction" dataset.
- Raw: 5,200 rows, 51 columns.
- Cleaned: 4,981 rows after removing 219 duplicate rows.
- Target: `prognosis`, 11 classes.
- Features: 45; 42 symptom/history indicators plus 3 weather variables
  (`Temperature (C)`, `Humidity`, `Wind Speed (km/h)`).
- Known property: every cleaned feature combination maps to exactly one label;
  the dataset is effectively deterministic by construction.

## Evaluation

Held-out test set (20% of cleaned data), stratified. The test set is **not**
used for hyperparameter tuning or model selection; those use a separate
validation split from the training data.

| Metric | Value |
|---|---:|
| Accuracy | 0.9789 |
| Macro precision | 0.9797 |
| Macro recall | 0.9793 |
| Macro F1 | 0.9793 |

### Cross-validation (5-fold stratified, training data)

| Metric | Mean | Std | 95% CI |
|---|---:|---:|---:|
| Accuracy | 0.9794 | 0.0035 | 0.9764–0.9825 |
| Macro F1 | 0.9790 | 0.0041 | 0.9754–0.9826 |

### Calibration (held-out test set)

| Metric | Value |
|---|---:|
| Brier score (multiclass) | 0.0405 |
| Expected calibration error | 0.0182 |
| Reliability diagram | `reports/reliability_diagram.png` |

### Fairness / error slices

| Slice | n | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Age <30 | 233 | 0.9700 | 0.9711 |
| Age 30–44 | 162 | 0.9877 | 0.9904 |
| Age 45–59 | 192 | 0.9948 | 0.9952 |
| Age 60+ | 410 | 0.9732 | 0.9700 |
| Gender 0 | 497 | 0.9799 | 0.9819 |
| Gender 1 | 500 | 0.9780 | 0.9754 |

Accuracy gap: 0.0248 across age bands, 0.0019 across gender. This is a
descriptive slice analysis, not a formal fairness certification.

### SHAP attribution (test sample)

Top mean absolute SHAP features: `headache`, `chest_pain`, `nausea`,
`vomiting`, `high_fever`, `Temperature (C)`, `dizziness`, `severe_headache`,
`rashes`, `Age`. See `reports/shap_summary.png` and
`reports/shap_feature_importance.csv`.

### Feature ablation (test set, same model family)

| Feature set | Features | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Weather only | 3 | 0.3761 | 0.2294 |
| Symptoms only | 42 | 0.9870 | 0.9880 |
| Full | 45 | 0.9799 | 0.9788 |

**Interpretation:** the weather variables carry little standalone signal.
The model is primarily learning symptom-to-disease patterns. This is a
property of the synthetic dataset, not evidence that weather drives disease.

## Limitations

- Synthetic and deterministic labels; metrics are optimistic for real-world use.
- No external or temporal validation.
- Calibration and slice metrics are descriptive; no formal fairness certification.
- The full feature set is slightly worse than symptoms-only; weather features
  may add noise. Test with a properly powered ablation before claiming value.
- Predictions are class labels; use the API's `confidence` and `top_k` fields
  rather than treating a label as certainty.

## Reproduce

```bash
uv run dvc repro
uv run -m src.evaluate
uv run -m src.ablation
uv run -m src.calibration
uv run -m src.slices
uv run -m src.cross_validate
uv run -m src.explain
```

## Versioning and traceability

- Data and artifacts: DVC (`dvc.lock`, `*.dvc`).
- Experiments and registry: MLflow (`@champion` alias).
- Serving exposes `model_version`; the deployment should set it from the
  registered model version.
