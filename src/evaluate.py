from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import MODEL_DIR
from src.interpret import (
    build_explanation_text,
    compute_model_importance_frame,
    compute_permutation_importance_frame,
    load_pickle,
    save_confusion_matrix_plot,
    save_importance_plot,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the saved student dropout model."
    )
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_DIR),
        help="Directory containing saved model artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    model_artifact = load_pickle(model_dir / "model.pkl")
    evaluation_data = load_pickle(model_dir / "evaluation_data.pkl")
    comparison_frame = pd.read_csv(model_dir / "model_comparison.csv")

    pipeline = model_artifact["pipeline"]
    X_test = evaluation_data["X_test"]
    y_test = evaluation_data["y_test"]

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    report_text = classification_report(y_test, predictions)
    report_dict = classification_report(y_test, predictions, output_dict=True)
    permutation_frame = compute_permutation_importance_frame(pipeline, X_test, y_test)
    model_importance_frame = compute_model_importance_frame(
        pipeline,
        model_artifact["feature_columns"],
    )
    explanation_text = build_explanation_text(
        model_artifact,
        comparison_frame,
        permutation_frame,
        model_importance_frame,
    )

    save_confusion_matrix_plot(confusion, model_dir / "confusion_matrix.png")
    save_importance_plot(
        permutation_frame,
        model_dir / "feature_importance.png",
        "Permutation Feature Importance",
        "importance_mean",
    )
    save_importance_plot(
        model_importance_frame,
        model_dir / "model_specific_feature_importance.png",
        "Model-Specific Feature Importance",
        "importance",
    )

    with (model_dir / "classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(report_text)

    permutation_frame.to_csv(model_dir / "feature_importance.csv", index=False)
    model_importance_frame.to_csv(
        model_dir / "model_specific_feature_importance.csv",
        index=False,
    )

    with (model_dir / "model_explanation.md").open("w", encoding="utf-8") as file:
        file.write(explanation_text)

    with (model_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model_name": model_artifact["model_name"],
                "accuracy": float(accuracy),
                "confusion_matrix": confusion.tolist(),
                "classification_report": report_dict,
            },
            file,
            indent=2,
        )

    print(f"Evaluated model: {model_artifact['model_name']}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report_text)
    print(f"Saved confusion matrix to: {model_dir / 'confusion_matrix.png'}")
    print(f"Saved permutation importance plot to: {model_dir / 'feature_importance.png'}")
    print(
        "Saved model-specific importance plot to: "
        f"{model_dir / 'model_specific_feature_importance.png'}"
    )
    print(f"Saved explanation report to: {model_dir / 'model_explanation.md'}")


if __name__ == "__main__":
    main()
