from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    DERIVED_FEATURES,
    MODEL_DIR,
    PRIMARY_APP_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.modeling import (
    build_model_comparison_frame,
    get_model_candidates,
    run_model_search,
)
from src.preprocess import build_default_values, build_input_ranges, load_dataset, split_features_and_target


def ensure_model_directory() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_pickle(file_path: Path, payload) -> None:
    with file_path.open("wb") as file:
        pickle.dump(payload, file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train student dropout prediction models."
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the dataset CSV file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_model_directory()

    dataframe = load_dataset(args.dataset)
    X, y = split_features_and_target(dataframe, target_column=TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    search_results = []
    trained_models = {}
    metrics_by_model = {}

    for model_name, config in get_model_candidates().items():
        result = run_model_search(
            model_name,
            config["estimator"],
            config["param_grid"],
            X_train,
            y_train,
            X_test,
            y_test,
        )
        search_results.append(result)
        trained_models[model_name] = result.best_pipeline
        metrics_by_model[model_name] = {
            **result.holdout_metrics,
            "model_name": model_name,
            "best_params": result.best_params,
            "cv_accuracy_mean": result.cv_accuracy_mean,
            "cv_accuracy_std": result.cv_accuracy_std,
        }

        print(
            f"{model_name} holdout accuracy: "
            f"{result.holdout_metrics['accuracy']:.4f} "
            f"(cv: {result.cv_accuracy_mean:.4f} +/- {result.cv_accuracy_std:.4f})"
        )

    comparison_frame = build_model_comparison_frame(search_results)
    best_model_name = max(
        metrics_by_model,
        key=lambda current_name: (
            metrics_by_model[current_name]["accuracy"],
            metrics_by_model[current_name]["cv_accuracy_mean"],
            metrics_by_model[current_name]["f1_score"],
        ),
    )
    best_pipeline = trained_models[best_model_name]

    best_artifact = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "metrics": metrics_by_model[best_model_name],
        "cv_accuracy_mean": metrics_by_model[best_model_name]["cv_accuracy_mean"],
        "cv_accuracy_std": metrics_by_model[best_model_name]["cv_accuracy_std"],
        "feature_columns": list(X.columns),
        "default_values": build_default_values(X_train),
        "input_ranges": build_input_ranges(X_train),
        "target_column": TARGET_COLUMN,
        "primary_app_features": PRIMARY_APP_FEATURES,
        "derived_features": DERIVED_FEATURES,
        "model_comparison": comparison_frame.to_dict(orient="records"),
    }

    save_pickle(MODEL_DIR / "model.pkl", best_artifact)
    save_pickle(MODEL_DIR / "logistic_regression.pkl", trained_models["logistic_regression"])
    save_pickle(MODEL_DIR / "decision_tree.pkl", trained_models["decision_tree"])
    save_pickle(
        MODEL_DIR / "evaluation_data.pkl",
        {
            "X_test": X_test,
            "y_test": y_test,
        },
    )

    comparison_frame.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_by_model, metrics_file, indent=2)

    print(f"Best model: {best_model_name}")
    print(f"Saved best model to: {MODEL_DIR / 'model.pkl'}")


if __name__ == "__main__":
    main()
