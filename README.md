# 🌦️ Weather Disease Prediction

This project aims to predict the likelihood of weather-sensitive diseases using machine learning. By analyzing historical climate and health records, it provides early warnings for disease outbreaks, empowering public health systems to respond proactively.

## 🧩 Problem Statement

Weather patterns influence the prevalence and spread of many diseases such as asthma, flu, and other respiratory conditions. The challenge is to build a robust prediction system that can:

- Accurately classify disease categories based on environmental conditions.
- Offer explainability of the predictions for healthcare stakeholders.
- Generalize well to unseen data from other regions or time periods.

## 🗃️ Dataset Overview

The dataset contains features related to weather and environmental measurements along with disease labels. Typical columns include:

| Feature         | Description                                      |
|-----------------|--------------------------------------------------|
| Temperature     | Daily average temperature in degrees Celsius     |
| Humidity        | Relative humidity percentage                     |
| Rainfall        | Precipitation in mm                              |
| WindSpeed       | Average wind speed in km/h                       |
| Pressure        | Atmospheric pressure in hPa                      |
| Gender          | Gender of the affected individual (if known)     |
| AgeGroup        | Age bracket of the patient                       |
| DiseaseLabel    | Target class (e.g., Asthma, Flu, Healthy, etc.)  |

## 📁 Project Structure

```
project1/
├── data/
│   ├── raw/               # Raw input datasets
│   ├── processed/         # Preprocessed train/test datasets
├── models/                # Trained model pipelines
├── evaluation/            # Model evaluation reports and plots
├── src/
│   ├── train.py           # Entry script for training and optimization
│   ├── evaluate.py        # Evaluation script
│   ├── utils/             # Preprocessing and utility functions
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/weather-disease-prediction.git
cd weather-disease-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the data

Place your raw dataset in the `data/raw/` directory. Ensure your preprocessing code outputs:

- `x_train.csv`, `y_train.csv`
- `x_test.csv`, `y_test.csv`
- `label_encoder.pkl`

These should be saved in `data/processed/`.

### 5. Train the model

```bash
python src/train.py
```

This saves the best model pipeline to `models/best_model.pkl`.

### 6. Evaluate the model

```bash
python src/evaluate.py
```

Outputs will be saved to the `evaluation/` directory, including:

- `evaluation_metrics.csv`
- `multiclass_roc_curve.png`
- `feature_importance.png`

---

## 🧠 Features

- End-to-end scikit-learn pipeline
- Hyperparameter optimization using Hyperopt
- Multiclass classification support
- Evaluation metrics and plots
- Feature importance for model interpretability

---

## 📊 Evaluation Outputs

| Metric     | Description                              |
|------------|------------------------------------------|
| Accuracy   | Overall correct predictions              |
| Precision  | Correctness among positive predictions   |
| Recall     | Coverage of actual positives             |
| F1-Score   | Harmonic mean of precision & recall      |

### 📈 ROC Curve

Saved to: `evaluation/multiclass_roc_curve.png`

### 📌 Feature Importance

Saved to: `evaluation/feature_importance.png`

---

## ⚙️ Configuration

Modify paths and logic in `WeatherDiseaseEvaluator` inside `evaluate.py` if needed.

---

## ✅ Requirements

- Python 3.8+
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy
- joblib
- hyperopt

```bash
pip install -r requirements.txt
```

---

## 🧪 Testing

To test your setup or add unit tests:

```bash
pytest tests/
```

---

## 📌 Notes

- Label encoding is required for correct ROC/metric computation.
- Only models with `.feature_importances_` are supported for feature explanation.
- SHAP and PDP (partial dependence plots) are excluded for simplicity and clarity.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙋🏽‍♀️ Contact

Created by [Your Name](mailto:your.email@example.com). Feel free to reach out with questions, issues, or suggestions.
