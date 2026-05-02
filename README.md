# Student Dropout Prediction

This project predicts whether a student is likely to drop out using supervised machine learning. It uses a modular preprocessing, feature-engineering, model-selection, and interpretation workflow, compares tuned Logistic Regression and Decision Tree models, saves the best model, and exposes predictions through a Streamlit app.

## Project Overview

The solution is built for an anonymized academic dataset stored in `student_dropout_dataset.csv`. It:

- loads and validates the dataset with pandas
- removes the irrelevant `student_id` column
- imputes missing values with mean and mode strategies
- encodes binary and multi-class categorical variables separately
- scales numerical features with `StandardScaler`
- creates derived features such as `performance_score`, `risk_score`, and `engagement_intensity`
- tunes and compares Logistic Regression and Decision Tree classifiers with cross-validation
- evaluates the selected model with accuracy, confusion matrix, and classification report
- explains the selected model with permutation importance and model-specific feature importance
- saves the best model to `models/model.pkl`
- provides a Streamlit interface for quick predictions

## Dataset Description

The dataset contains student-level academic, engagement, and background features, including:

- demographics and family context
- academic performance and subject failures
- attendance, participation, and assignment behavior
- platform usage and engagement indicators
- counseling and stress-related attributes

Target column:

- `dropout_status`

This project treats the data as anonymized and does not expose personal identifiers in the pipeline or app.

## Project Structure

```text
Student-Dropout-Prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── student_dropout_dataset.csv
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── config.py
    ├── features.py
    ├── modeling.py
    ├── interpret.py
    ├── preprocess.py
    ├── train.py
    └── evaluate.py
```

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

Train the models and save artifacts:

```bash
python -m src.train
```

Evaluate the saved model and generate plots:

```bash
python -m src.evaluate
```

Launch the Streamlit app:

```bash
streamlit run app.py
```

## Results Explanation

- `src.train` prints model accuracies and saves the best-performing model.
- `src.evaluate` generates:
  - `models/confusion_matrix.png`
  - `models/feature_importance.png`
  - `models/model_specific_feature_importance.png`
  - `models/feature_importance.csv`
  - `models/model_explanation.md`
  - `models/model_comparison.csv`
  - `models/classification_report.txt`
  - `models/evaluation_metrics.json`
- `app.py` allows quick predictions by accepting attendance, GPA, failures, and engagement as inputs while filling the remaining features from training defaults.

## Improvements To Consider

- add Random Forest for stronger non-linear performance
- add XGBoost or LightGBM for boosted-tree benchmarking
- run cross-validation and hyperparameter tuning
- add model monitoring and drift checks
- deploy the Streamlit app on Streamlit Community Cloud, Azure, or Docker
