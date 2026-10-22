"""Benchmark two nonlinear families against the existing Logistic Regression.

Run from the project root:
    .venv\\Scripts\\python.exe src/benchmark_advanced_models.py

Fixed exploratory configurations, not an outer-fold hyperparameter search.
Every model uses the same five stratified folds. Preprocessing is fitted only
on each outer training fold; early stopping uses an internal validation split.
Existing predictors and saved model files are not changed by this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from career_ml import FEATURE_COLUMNS, TARGET_COLUMN, load_training_data
from evaluation import metric_values, summarize_folds
from experiment_models import MODEL_NAMES, make_experiment, describe_preprocessing


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/raw/Career_Path_Dataset_with_Career.csv"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--models", choices=MODEL_NAMES, nargs="+", default=MODEL_NAMES)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def write_reports(report_dir, metadata, results, fold_rows):
    """Write progress after each completed fold for inspectable long runs."""
    (report_dir / "advanced_benchmark.json").write_text(
        json.dumps({**metadata, "models": results}, indent=2), encoding="utf-8"
    )
    pd.DataFrame(fold_rows).to_csv(report_dir / "advanced_fold_metrics.csv", index=False)
    summaries = [value["summary"] for value in results.values() if "summary" in value]
    if summaries:
        pd.DataFrame(summaries).to_csv(report_dir / "advanced_model_comparison.csv", index=False)


def main():
    args = parse_args()
    data = load_training_data(args.data)
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].astype(str)
    labels = np.sort(y.unique())
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows": len(data),
        "input_features": FEATURE_COLUMNS,
        "classes": labels.tolist(),
        "class_counts": y.value_counts().sort_index().to_dict(),
        "cross_validation": {
            "splitter": "StratifiedKFold", "n_splits": 5,
            "shuffle": True, "random_state": 42,
            "preprocessing": "Fit independently on each outer training fold",
            "early_stopping": "Internal training-fold validation; no outer test access",
            "standard_deviation": "Population standard deviation across five folds (ddof=0)",
        },
        "versions": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
        "native_thread_limit": args.threads,
        "scope": "Fixed exploratory configurations; not optimized or a significance test",
        "full_data_models_saved": False,
        "status": "running",
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    fold_rows = []
    with threadpool_limits(limits=args.threads):
        for model_name in dict.fromkeys(args.models):
            preprocess, model, description = make_experiment(model_name)
            results[model_name] = {
                **description, "parameters": model.get_params(), "folds": [],
                "preprocessing_parameters": describe_preprocessing(preprocess),
                "numeric_encoding": "Training-fold median imputation and StandardScaler",
            }
            oof_rows = []
            started = time.perf_counter()
            for fold_number, (train_indices, test_indices) in enumerate(folds, start=1):
                preprocess, model, _ = make_experiment(model_name)
                print(f"{model_name}: fold {fold_number}/5 starting", flush=True)
                fold_started = time.perf_counter()
                train_X = preprocess.fit_transform(X.iloc[train_indices], y.iloc[train_indices])
                test_X = preprocess.transform(X.iloc[test_indices])
                preprocess_seconds = time.perf_counter() - fold_started
                fit_started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(train_X, y.iloc[train_indices])
                fit_seconds = time.perf_counter() - fit_started
                probabilities = model.predict_proba(test_X)
                predictions = model.classes_[np.argmax(probabilities, axis=1)]
                scores = metric_values(y.iloc[test_indices], predictions, probabilities, model.classes_)
                n_iter = getattr(model, "n_iter_", None)
                iterations = int(np.max(n_iter)) if n_iter is not None else None
                fold_result = {
                    "model": model_name, "fold": fold_number,
                    "train_rows": len(train_indices), "test_rows": len(test_indices),
                    "encoded_features": train_X.shape[1], **scores,
                    "preprocess_seconds": preprocess_seconds, "fit_seconds": fit_seconds,
                    "total_seconds": time.perf_counter() - fold_started,
                    "iterations": iterations,
                    "hit_iteration_limit": iterations == model.max_iter,
                    "warnings": [str(item.message) for item in caught],
                }
                if model_name == "mlp_neural_network":
                    fold_result["best_internal_validation_accuracy"] = float(model.best_validation_score_)
                    fold_result["final_training_loss"] = float(model.loss_)
                elif model_name == "hist_gradient_boosting":
                    fold_result["final_internal_validation_negative_loss"] = float(model.validation_score_[-1])
                results[model_name]["folds"].append(fold_result)
                fold_rows.append(fold_result)
                # Match sklearn's top-k tie convention (higher class index first).
                order = np.argsort(probabilities, axis=1, kind="stable")[:, ::-1]
                ranked_labels = model.classes_[order]
                for row_number, source_index in enumerate(test_indices):
                    truth = y.iloc[source_index]
                    row = {
                        "source_row_index": int(source_index), "fold": fold_number,
                        "actual_career": truth,
                        "actual_rank": int(np.flatnonzero(ranked_labels[row_number] == truth)[0]) + 1,
                        "top_1_prediction": predictions[row_number],
                    }
                    for k in (1, 2, 3, 4):
                        row[f"rank_{k}_career"] = ranked_labels[row_number, k - 1]
                        row[f"rank_{k}_probability"] = float(probabilities[row_number, order[row_number, k - 1]])
                    oof_rows.append(row)
                print(
                    f"{model_name}: fold {fold_number}/5 "
                    f"top-1={scores['accuracy']:.4f} top-3={scores['top_3_accuracy']:.4f} "
                    f"macro-F1={scores['macro_f1']:.4f} iterations={iterations} "
                    f"seconds={fold_result['total_seconds']:.1f}", flush=True,
                )
                write_reports(args.report_dir, metadata, results, fold_rows)
            summary = {"model": model_name, "cv_folds": len(folds)}
            summary.update(summarize_folds(results[model_name]["folds"], scores, prefix="cv_"))
            summary["cv_seconds"] = time.perf_counter() - started
            results[model_name]["summary"] = summary
            pd.DataFrame(oof_rows).sort_values("source_row_index").to_csv(
                args.report_dir / f"advanced_{model_name}_oof.csv", index=False
            )
            write_reports(args.report_dir, metadata, results, fold_rows)
            print(f"Completed {model_name}: {json.dumps(summary)}", flush=True)
    metadata["status"] = "complete"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_reports(args.report_dir, metadata, results, fold_rows)


if __name__ == "__main__":
    main()
