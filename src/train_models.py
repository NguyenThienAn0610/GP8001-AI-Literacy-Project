"""Train and compare the three selected career-path classifiers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from career_ml import (
    FEATURE_COLUMNS,
    MODEL_FACTORIES,
    TARGET_COLUMN,
    make_pipeline,
    load_training_data,
)
from evaluation import metric_values, summarize_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/Career_Path_Dataset_with_Career.csv"),
        help="Labelled CSV used for training.",
    )
    model_selection = parser.add_mutually_exclusive_group()
    model_selection.add_argument(
        "--model",
        choices=list(MODEL_FACTORIES),
        help="Train one specific model.",
    )
    model_selection.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_FACTORIES),
        help="Train two or more selected models.",
    )
    model_selection.add_argument(
        "--all-models",
        action="store_true",
        help="Train all available models (also the default when no selection is given).",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def selected_model_names(args: argparse.Namespace) -> list[str]:
    """Resolve the mutually exclusive model-selection CLI options."""
    if args.model:
        return [args.model]
    if args.models:
        # Preserve the user's order while avoiding accidental duplicate training.
        return list(dict.fromkeys(args.models))
    return list(MODEL_FACTORIES)


def main() -> None:
    args = parse_args()
    model_names = selected_model_names(args)
    print(f"Selected model(s): {', '.join(model_names)}", flush=True)
    data = load_training_data(args.data)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].astype(str)
    folds = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, float | str]] = []

    for model_name in model_names:
        print(f"Training {model_name} ...", flush=True)
        evaluation_pipeline = make_pipeline(model_name)
        started = time.perf_counter()
        predictions = pd.Series(index=y.index, dtype="object")
        fold_scores = []

        for train_indices, test_indices in folds.split(X, y):
            fold_pipeline = clone(evaluation_pipeline)
            fold_pipeline.fit(X.iloc[train_indices], y.iloc[train_indices])
            fold_predictions = fold_pipeline.predict(X.iloc[test_indices])
            fold_probabilities = fold_pipeline.predict_proba(X.iloc[test_indices])
            fold_labels = fold_pipeline.classes_
            predictions.iloc[test_indices] = fold_predictions
            fold_scores.append(metric_values(
                y.iloc[test_indices], fold_predictions, fold_probabilities, fold_labels,
            ))

        cv_seconds = time.perf_counter() - started

        row = {
            "model": model_name,
            "cv_folds": args.cv_folds,
            **summarize_folds(fold_scores, fold_scores[0], prefix="cv_"),
            "cv_seconds": cv_seconds,
        }
        metrics.append(row)

        report = classification_report(y, predictions, output_dict=True)
        report_path = args.report_dir / f"{model_name}_classification_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"Fitting {model_name} on all {len(X):,} rows ...", flush=True)
        pipeline = make_pipeline(model_name)
        refit_started = time.perf_counter()
        pipeline.fit(X, y)
        row["full_fit_seconds"] = time.perf_counter() - refit_started

        model_path = args.model_dir / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path, compress=3)
        print(
            f"Saved {model_path} | mean CV macro-F1={row['cv_macro_f1_mean']:.4f} | "
            f"top-1={row['cv_accuracy_mean']:.4f} | "
            f"top-3={row['cv_top_3_accuracy_mean']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(metrics).sort_values("cv_macro_f1_mean", ascending=False)
    comparison.to_csv(args.report_dir / "model_comparison.csv", index=False)
    print(f"\n{args.cv_folds}-fold cross-validation (higher is better):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
