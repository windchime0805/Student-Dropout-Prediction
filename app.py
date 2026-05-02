from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/model.pkl")
FEATURE_IMPORTANCE_PATH = Path("models/feature_importance.csv")
PRIMARY_INPUTS = [
    "attendance_rate",
    "gpa",
    "subject_failure_count",
    "engagement_score",
]


@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model not found. Run `python -m src.train` first."
        )

    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def build_input_frame(default_values, overrides):
    payload = default_values.copy()
    payload.update(overrides)
    return pd.DataFrame([payload])


@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return None


def render_numeric_input(label, config, is_float=False):
    minimum = config["min"]
    maximum = config["max"]
    default = config["default"]

    if is_float:
        step = 0.01
        return st.number_input(
            label,
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(default),
            step=step,
            format="%.2f",
        )

    return st.number_input(
        label,
        min_value=int(minimum),
        max_value=int(maximum),
        value=int(default),
        step=1,
    )


def main():
    st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
    st.title("Student Dropout Prediction")
    st.write(
        "Provide a few student performance signals to estimate dropout risk. "
        "The remaining features are filled with training defaults for a quick screen."
    )

    artifact = load_model_artifact()
    pipeline = artifact["pipeline"]
    default_values = artifact["default_values"]
    input_ranges = artifact["input_ranges"]
    feature_importance = load_feature_importance()

    st.caption(
        f"Best model: {artifact['model_name']} | "
        f"Holdout accuracy: {artifact['metrics']['accuracy']:.4f} | "
        f"CV accuracy: {artifact['cv_accuracy_mean']:.4f}"
    )

    attendance_rate = render_numeric_input(
        "Attendance Rate (%)",
        input_ranges["attendance_rate"],
    )
    gpa = render_numeric_input(
        "GPA",
        input_ranges["gpa"],
        is_float=True,
    )
    subject_failure_count = render_numeric_input(
        "Subject Failure Count",
        input_ranges["subject_failure_count"],
    )
    engagement_score = render_numeric_input(
        "Engagement Score",
        input_ranges["engagement_score"],
    )

    if st.button("Predict Dropout Status"):
        overrides = {
            "attendance_rate": attendance_rate,
            "gpa": gpa,
            "subject_failure_count": subject_failure_count,
            "engagement_score": engagement_score,
        }
        input_frame = build_input_frame(default_values, overrides)
        prediction = int(pipeline.predict(input_frame)[0])

        probability_text = ""
        if hasattr(pipeline, "predict_proba"):
            probability = float(pipeline.predict_proba(input_frame)[0][1])
            probability_text = f"Estimated dropout probability: {probability:.2%}"

        if prediction == 1:
            st.error("Prediction: The student is at risk of dropping out.")
        else:
            st.success("Prediction: The student is unlikely to drop out.")

        if probability_text:
            st.write(probability_text)

        if feature_importance is not None and not feature_importance.empty:
            st.write("Top global features used by the model:")
            st.dataframe(feature_importance.head(5), use_container_width=True)

        st.caption(
            "This tool supports screening and should complement advisor or academic "
            "support decisions rather than replace them."
        )


if __name__ == "__main__":
    main()
