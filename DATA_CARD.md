# Data Card: Weather-Related Disease Prediction

## Summary

| Item | Value |
|---|---|
| Raw rows | 5,200 |
| Columns | 51 |
| Cleaned rows | 4,981 |
| Duplicate rows removed | 219 |
| Target | `prognosis` |
| Classes | 11 |
| Features used | 45 |

## Provenance

The dataset is the public **"Weather-related disease prediction"** dataset. It
appears to be synthetically generated: every cleaned feature combination maps to
exactly one disease label. No personally identifiable information is present.

The raw CSV is tracked by DVC (`data/raw/*.dvc`). Cleaning, splitting, and
feature artifacts are DVC pipeline outputs (`dvc.lock`).

## Intended use

Education and portfolio demonstration of an end-to-end MLOps system. Not for
clinical diagnosis, triage, or any health decision.

## Features

- **Weather:** `Temperature (C)`, `Humidity`, `Wind Speed (km/h)`
- **Demographics:** `Age`, `Gender`
- **Symptoms/history:** 40 binary indicators such as `headache`, `chest_pain`,
  `nausea`, `high_fever`, `joint_pain`, `rashes`, `cough`, and `fatigue`.

Columns dropped during cleaning: `shivering`, `asthma_history`,
`high_cholesterol`, `diabetes`, `obesity`.

## Target distribution (cleaned)

| Class | Rows |
|---|---:|
| Heart Attack | 968 |
| Migraine | 897 |
| Influenza | 632 |
| Heat Stroke | 323 |
| Malaria | 319 |
| Stroke | 312 |
| Eczema | 311 |
| Common Cold | 309 |
| Dengue | 308 |
| Sinusitis | 301 |
| Arthritis | 301 |

## Quality checks

`src/validate_data.py` enforces:

- non-empty dataset and target present
- no missing values
- binary feature values are in `{0, 1}`
- age/temperature/humidity/wind ranges
- at least two classes, with at least two rows per class
- no feature combination with conflicting labels

Current status: **passed** (`reports/data_validation.json`).

## Known limitations

- Deterministic synthetic labels; high metrics reflect the generator, not
  clinical signal.
- Weather features have little standalone predictive value (weather-only
  accuracy 0.3761).
- No missingness, temporal ordering, or external validation.
- Class imbalance is moderate (largest class 968, smallest 301).
- No subgroup labels beyond age and gender.

## Ethical considerations

Do not present this model as a diagnostic system. Any real deployment would
require clinical review, external validation, calibration under shift, privacy
review, and regulatory assessment.
