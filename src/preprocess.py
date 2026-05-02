from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import ID_COLUMN, PRIMARY_APP_FEATURES, TARGET_COLUMN
from src.features import FeatureEngineer


def to_python_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


class BinaryLabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """Apply a separate LabelEncoder to each binary categorical feature."""

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        frame = self._to_frame(X)
        self.encoders_ = {}

        for column in self.feature_names:
            encoder = LabelEncoder()
            values = frame[column].astype(str)
            if "Unknown" not in values.values:
                values = pd.concat([values, pd.Series(["Unknown"])], ignore_index=True)
            encoder.fit(values)
            self.encoders_[column] = encoder

        return self

    def transform(self, X):
        frame = self._to_frame(X)
        encoded_columns = [
            self.encoders_[column].transform(
                frame[column]
                .astype(str)
                .where(
                    frame[column].astype(str).isin(self.encoders_[column].classes_),
                    "Unknown",
                )
            )
            for column in self.feature_names
        ]
        return np.column_stack(encoded_columns)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names, dtype=object)

    def _to_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names].copy()

        return pd.DataFrame(X, columns=self.feature_names)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    return pd.read_csv(dataset_path)


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    feature_frame = dataframe.drop(columns=[target_column]).copy()
    if ID_COLUMN in feature_frame.columns:
        feature_frame = feature_frame.drop(columns=[ID_COLUMN])

    target_series = dataframe[target_column].copy()
    return feature_frame, target_series


def detect_feature_types(
    feature_frame: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    categorical_features = feature_frame.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_features = [
        column for column in feature_frame.columns if column not in categorical_features
    ]
    binary_features = [
        column
        for column in categorical_features
        if feature_frame[column].dropna().nunique() == 2
    ]
    multiclass_features = [
        column
        for column in categorical_features
        if feature_frame[column].dropna().nunique() > 2
    ]

    return numeric_features, binary_features, multiclass_features


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_features, binary_features, multiclass_features = detect_feature_types(
        feature_frame
    )
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if binary_features:
        binary_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    BinaryLabelEncoderTransformer(feature_names=binary_features),
                ),
            ]
        )
        transformers.append(("binary", binary_pipeline, binary_features))

    if multiclass_features:
        multiclass_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("multiclass", multiclass_pipeline, multiclass_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_default_values(feature_frame: pd.DataFrame) -> dict:
    default_values = {}

    for column in feature_frame.columns:
        series = feature_frame[column]
        if pd.api.types.is_numeric_dtype(series):
            default_values[column] = to_python_scalar(series.median())
        else:
            default_values[column] = to_python_scalar(series.mode(dropna=True).iloc[0])

    return default_values


def build_input_ranges(feature_frame: pd.DataFrame) -> dict:
    input_ranges = {}

    for column in PRIMARY_APP_FEATURES:
        series = feature_frame[column]
        input_ranges[column] = {
            "min": to_python_scalar(series.min()),
            "max": to_python_scalar(series.max()),
            "default": to_python_scalar(series.median()),
        }

    return input_ranges
