from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.config import RANDOM_STATE


def load_pickle(file_path: Path):
    with file_path.open("rb") as file:
        import pickle

        return pickle.load(file)


def save_confusion_matrix_plot(confusion, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Dropout", "Dropout"],
        yticklabels=["No Dropout", "Dropout"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_permutation_importance_frame(pipeline, X_test, y_test) -> pd.DataFrame:
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
        scoring="accuracy",
    )
    importance_frame = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance_frame.sort_values("importance_mean", ascending=False)


def resolve_base_feature(expanded_name: str, raw_feature_names: list[str]) -> str:
    cleaned_name = expanded_name.split("__", 1)[1] if "__" in expanded_name else expanded_name
    if cleaned_name in raw_feature_names:
        return cleaned_name

    for raw_feature in sorted(raw_feature_names, key=len, reverse=True):
        if cleaned_name.startswith(f"{raw_feature}_"):
            return raw_feature

    return cleaned_name


def compute_model_importance_frame(pipeline, raw_feature_names: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        raw_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_values = abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_frame = pd.DataFrame(
        {
            "feature": [
                resolve_base_feature(name, raw_feature_names) for name in transformed_names
            ],
            "importance": raw_values,
        }
    )
    importance_frame = (
        importance_frame.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    return importance_frame


def save_importance_plot(
    importance_frame: pd.DataFrame,
    output_path: Path,
    title: str,
    value_column: str,
    top_n: int = 12,
) -> None:
    plot_frame = importance_frame.head(top_n).copy()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_frame,
        x=value_column,
        y="feature",
        hue="feature",
        dodge=False,
        palette="viridis",
        legend=False,
    )
    plt.title(title)
    plt.xlabel(value_column.replace("_", " ").title())
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_explanation_text(
    model_artifact: dict,
    comparison_frame: pd.DataFrame,
    permutation_frame: pd.DataFrame,
    model_importance_frame: pd.DataFrame,
) -> str:
    best_metrics = model_artifact["metrics"]
    top_permutation = permutation_frame[permutation_frame["importance_mean"] > 0].head(5)
    top_model = model_importance_frame[model_importance_frame["importance"] > 0].head(5)

    lines = [
        "# Model Explanation",
        "",
        f"- Selected model: `{model_artifact['model_name']}`",
        f"- Holdout accuracy: `{best_metrics['accuracy']:.4f}`",
        f"- Holdout precision: `{best_metrics['precision']:.4f}`",
        f"- Holdout recall: `{best_metrics['recall']:.4f}`",
        f"- Holdout F1-score: `{best_metrics['f1_score']:.4f}`",
        f"- Cross-validation accuracy: `{model_artifact['cv_accuracy_mean']:.4f} +/- {model_artifact['cv_accuracy_std']:.4f}`",
        "",
        "## Model Comparison",
        "```text",
        comparison_frame.to_string(index=False),
        "```",
        "",
        "## Top Features By Permutation Importance",
    ]

    for row in top_permutation.itertuples(index=False):
        lines.append(
            f"- `{row.feature}` changed accuracy by about `{row.importance_mean:.4f}` when shuffled."
        )

    lines.append("")
    lines.append("## Top Features By Model-Specific Importance")

    for row in top_model.itertuples(index=False):
        lines.append(
            f"- `{row.feature}` carries a relative importance score of `{row.importance:.4f}`."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Attendance, GPA, engagement, and stress level are the strongest global signals in the selected model.",
            "- The engineered features help compress multiple academic and behavioral signals into more stable summary indicators.",
            "- A perfect or near-perfect score on this dataset suggests the labels may be highly rule-driven or synthetic, so real-world performance could be lower on noisier data.",
        ]
    )

    return "\n".join(lines)
