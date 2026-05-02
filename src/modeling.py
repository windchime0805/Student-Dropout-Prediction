from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_FOLDS, RANDOM_STATE
from src.features import FeatureEngineer
from src.preprocess import build_preprocessor


@dataclass
class ModelSearchResult:
    model_name: str
    best_pipeline: Pipeline
    best_params: dict
    cv_accuracy_mean: float
    cv_accuracy_std: float
    holdout_metrics: dict


def build_training_pipeline(feature_frame: pd.DataFrame, estimator) -> Pipeline:
    engineered_sample = FeatureEngineer().fit_transform(feature_frame.copy())
    preprocessor = build_preprocessor(engineered_sample)

    return Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def get_model_candidates() -> dict:
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "param_grid": {
                "model__C": [0.5, 1.0, 2.0, 5.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "model__max_depth": [4, 5, 6, 8, None],
                "model__min_samples_leaf": [20, 10, 5, 1],
                "model__class_weight": [None, "balanced"],
            },
        },
    }


def evaluate_predictions(y_true, predictions) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def run_model_search(
    model_name: str,
    estimator,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelSearchResult:
    pipeline = build_training_pipeline(X_train, estimator)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    predictions = search.best_estimator_.predict(X_test)
    holdout_metrics = evaluate_predictions(y_test, predictions)

    return ModelSearchResult(
        model_name=model_name,
        best_pipeline=search.best_estimator_,
        best_params=search.best_params_,
        cv_accuracy_mean=float(search.best_score_),
        cv_accuracy_std=float(
            search.cv_results_["std_test_score"][search.best_index_]
        ),
        holdout_metrics=holdout_metrics,
    )


def build_model_comparison_frame(results: list[ModelSearchResult]) -> pd.DataFrame:
    records = []

    for result in results:
        records.append(
            {
                "model_name": result.model_name,
                "cv_accuracy_mean": result.cv_accuracy_mean,
                "cv_accuracy_std": result.cv_accuracy_std,
                "holdout_accuracy": result.holdout_metrics["accuracy"],
                "holdout_precision": result.holdout_metrics["precision"],
                "holdout_recall": result.holdout_metrics["recall"],
                "holdout_f1_score": result.holdout_metrics["f1_score"],
                "best_params": str(result.best_params),
            }
        )

    return pd.DataFrame(records).sort_values(
        by=["holdout_accuracy", "cv_accuracy_mean", "holdout_f1_score"],
        ascending=False,
    )
