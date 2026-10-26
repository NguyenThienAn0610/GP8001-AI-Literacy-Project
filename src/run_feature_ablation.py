"""Measure model performance after excluding one or more input features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

from career_ml import (
    FEATURE_COLUMNS,
    MODEL_FACTORIES,
    TARGET_COLUMN,
    make_pipeline,
    validate_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/Career_Path_Dataset_with_Career.csv"),
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_FACTORIES),
        default="logistic_regression",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        choices=FEATURE_COLUMNS,
        default=["Fields_of_Interest"],
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/fields_of_interest_ablation.csv"),
    )
    parser.add_argument(
        "--classification-report",
        type=Path,
        default=Path("reports/fields_of_interest_ablation_classification_report.json"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/logistic_regression_without_fields_of_interest.joblib"),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("reports/model_comparison.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")

    data = pd.read_csv(args.data)
    validate_features(data)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Training data must contain '{TARGET_COLUMN}'.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"'{TARGET_COLUMN}' contains missing labels.")

    excluded = list(dict.fromkeys(args.exclude))
    selected_features = [
        column for column in FEATURE_COLUMNS if column not in set(excluded)
    ]
    if not selected_features:
        raise ValueError("The ablation must retain at least one feature.")

    X = data[selected_features]
    y = data[TARGET_COLUMN].astype(str)
    folds = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )
    pipeline = make_pipeline(args.model, excluded_columns=excluded)

    predictions = pd.Series(index=y.index, dtype="object")
    accuracy: list[float] = []
    macro_f1: list[float] = []
    weighted_f1: list[float] = []
    top_2: list[float] = []
    top_3: list[float] = []
    top_4: list[float] = []
    started = time.perf_counter()

    print(
        f"Evaluating {args.model} without {', '.join(excluded)} "
        f"using {args.cv_folds}-fold cross-validation ...",
        flush=True,
    )
    for fold_number, (train_indices, test_indices) in enumerate(
        folds.split(X, y), start=1
    ):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X.iloc[train_indices], y.iloc[train_indices])
        fold_predictions = fold_pipeline.predict(X.iloc[test_indices])
        fold_probabilities = fold_pipeline.predict_proba(X.iloc[test_indices])
        labels = fold_pipeline.classes_
        predictions.iloc[test_indices] = fold_predictions

        fold_y = y.iloc[test_indices]
        accuracy.append(accuracy_score(fold_y, fold_predictions))
        macro_f1.append(f1_score(fold_y, fold_predictions, average="macro"))
        weighted_f1.append(
            f1_score(fold_y, fold_predictions, average="weighted")
        )
        top_2.append(
            top_k_accuracy_score(fold_y, fold_probabilities, k=2, labels=labels)
        )
        top_3.append(
            top_k_accuracy_score(fold_y, fold_probabilities, k=3, labels=labels)
        )
        top_4.append(
            top_k_accuracy_score(fold_y, fold_probabilities, k=4, labels=labels)
        )
        print(f"Completed fold {fold_number}/{args.cv_folds}", flush=True)

    result: dict[str, object] = {
        "model": args.model,
        "excluded_features": ",".join(excluded),
        "retained_feature_count": len(selected_features),
        "cv_folds": args.cv_folds,
        "cv_accuracy_mean": np.mean(accuracy),
        "cv_accuracy_std": np.std(accuracy),
        "cv_macro_f1_mean": np.mean(macro_f1),
        "cv_macro_f1_std": np.std(macro_f1),
        "cv_weighted_f1_mean": np.mean(weighted_f1),
        "cv_weighted_f1_std": np.std(weighted_f1),
        "cv_top_2_accuracy_mean": np.mean(top_2),
        "cv_top_2_accuracy_std": np.std(top_2),
        "cv_top_3_accuracy_mean": np.mean(top_3),
        "cv_top_3_accuracy_std": np.std(top_3),
        "cv_top_4_accuracy_mean": np.mean(top_4),
        "cv_top_4_accuracy_std": np.std(top_4),
        "cv_seconds": time.perf_counter() - started,
    }

    if args.baseline_report.exists():
        baseline_table = pd.read_csv(args.baseline_report)
        baseline_rows = baseline_table.loc[baseline_table["model"] == args.model]
        if not baseline_rows.empty:
            baseline = baseline_rows.iloc[0]
            for metric in [
                "cv_accuracy_mean",
                "cv_macro_f1_mean",
                "cv_weighted_f1_mean",
                "cv_top_2_accuracy_mean",
                "cv_top_3_accuracy_mean",
                "cv_top_4_accuracy_mean",
            ]:
                if metric in baseline:
                    result[f"baseline_{metric}"] = baseline[metric]
                    result[f"delta_{metric}"] = result[metric] - baseline[metric]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(args.report, index=False)

    args.classification_report.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(y, predictions, output_dict=True)
    args.classification_report.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Fitting final model on all {len(X):,} rows ...", flush=True)
    fit_started = time.perf_counter()
    pipeline.fit(X, y)
    result["full_fit_seconds"] = time.perf_counter() - fit_started
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_output, compress=3)

    # Rewrite the one-row report with the final-fit timing included.
    pd.DataFrame([result]).to_csv(args.report, index=False)
    print(pd.DataFrame([result]).to_string(index=False), flush=True)
    print(f"Saved report to {args.report}", flush=True)
    print(f"Saved fitted model to {args.model_output}", flush=True)


if __name__ == "__main__":
    main()
