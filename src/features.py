from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create derived features that capture performance and dropout risk."""

    performance_map = {"Low": 1, "Medium": 2, "High": 3}
    stress_map = {"Low": 1, "Medium": 2, "High": 3}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None
        return self

    def transform(self, X):
        frame = self._to_frame(X).copy()

        prior_performance = (
            frame["previous_academic_performance"]
            .map(self.performance_map)
            .fillna(2)
        )
        stress_signal = frame["stress_level"].map(self.stress_map).fillna(2)

        frame["performance_score"] = np.clip(
            (
                (frame["gpa"] / 4.0) * 0.35
                + (frame["attendance_rate"] / 100.0) * 0.15
                + (frame["assignment_submission_rate"] / 100.0) * 0.20
                + (frame["quiz_completion_rate"] / 100.0) * 0.15
                + (prior_performance / 3.0) * 0.15
            )
            * 100,
            0,
            100,
        )

        frame["risk_score"] = np.clip(
            frame["subject_failure_count"] * 8
            + (100 - frame["attendance_rate"]) * 0.30
            + (10 - frame["engagement_score"]) * 3.50
            + stress_signal * 6,
            0,
            100,
        )

        frame["engagement_intensity"] = np.clip(
            (
                frame["attendance_rate"] * 0.25
                + frame["class_participation_score"] * 3.0
                + frame["assignment_submission_rate"] * 0.20
                + frame["quiz_completion_rate"] * 0.20
                + frame["login_frequency"] * 0.80
                + frame["peer_interaction_score"] * 1.50
            ),
            0,
            100,
        )

        return frame

    def _to_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X

        if self.feature_names_in_ is None:
            raise ValueError("FeatureEngineer requires DataFrame inputs during fitting.")

        return pd.DataFrame(X, columns=self.feature_names_in_)
