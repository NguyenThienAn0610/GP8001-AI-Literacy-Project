"""Measure predictive information added by fields other than stated interest.

Run from project root: .venv\\Scripts\\python.exe src/analyze_feature_ablation.py
Uses the existing fixed model settings and the same five stratified folds.
No hyperparameter search or deployment model changes are made.
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
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from experiment_models import make_experiment
from evaluation import summarize_folds
from career_ml import FEATURE_COLUMNS, PROFILE_COLUMNS, TARGET_COLUMN, load_training_data


CONDITIONS = (
    "logistic_regression_all_features",
    "logistic_regression_interest_only",
    "logistic_regression_without_interest",
    "hist_gradient_boosting_without_interest",
)
INTEREST = "Fields_of_Interest"


def make_condition(condition):
    family = "hist_gradient_boosting" if condition.startswith("hist_") else "logistic_regression"
    preprocess, model, description = make_experiment(family)
    # Change selectors before fitting; excluded columns cannot enter tokenization,
    # vocabulary building, bigrams, or numerical preprocessing.
    if condition.endswith("interest_only"):
        preprocess.transformers = [
            (name, transformer, [INTEREST])
            for name, transformer, _ in preprocess.transformers if name == "profile"
        ]
    elif condition.endswith("without_interest"):
        preprocess.transformers = [
            (name, transformer, [column for column in columns if column != INTEREST])
            for name, transformer, columns in preprocess.transformers
        ]
    selected = [column for _, _, columns in preprocess.transformers for column in columns]
    assert INTEREST in PROFILE_COLUMNS
    assert len(selected) == (1 if condition.endswith("interest_only") else 17 if condition.endswith("without_interest") else 18)
    if condition.endswith("without_interest"):
        assert INTEREST not in selected
    profile = next(transformer for name, transformer, _ in preprocess.transformers if name == "profile")
    encoding = {
        "selectors": {name: columns for name, _, columns in preprocess.transformers},
        "vectorizer_parameters": {
            key: value for key, value in profile["tfidf"].get_params().items()
            if key not in ("tokenizer", "dtype")
        },
        "tokenizer": "str.split",
        "profile_transformer": "career_ml.ProfileTextTransformer",
        "sparse_threshold": preprocess.sparse_threshold,
        "numeric_encoding": "Median imputation and StandardScaler" if "Age" in selected else None,
    }
    return preprocess, model, {**description, "parameters": model.get_params(), "preprocessing": encoding}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/Career_Path_Dataset_with_Career.csv"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    data = load_training_data(args.data)
    X, y = data[FEATURE_COLUMNS], data[TARGET_COLUMN].astype(str)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "data_path": str(args.data),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows": len(data),
        "classes": sorted(y.unique().tolist()),
        "class_counts": y.value_counts().sort_index().to_dict(),
        "cross_validation": {
            "splitter": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42,
            "same_folds_for_all_conditions": True,
            "preprocessing": "Fitted only on each outer training fold",
            "early_stopping": "HGB internal validation only from outer training fold",
            "standard_deviation": "Population standard deviation of fold metrics (ddof=0)",
            "top_1_ties": "model.predict uses lowest class index",
            "top_3_ties": "sklearn top_k_accuracy_score uses highest class indices first",
        },
        "native_thread_limit": args.threads,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "scope": "Fixed settings; incremental prediction on this dataset, not causality or proof of independence",
        "conditions": {},
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    fold_rows, summary_rows = [], []

    def save():
        (args.report_dir / "feature_ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if fold_rows:
            pd.DataFrame(fold_rows).to_csv(args.report_dir / "feature_ablation_fold_metrics.csv", index=False)
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(args.report_dir / "feature_ablation_comparison.csv", index=False)

    with threadpool_limits(limits=args.threads):
        for condition in CONDITIONS:
            _, _, description = make_condition(condition)
            report["conditions"][condition] = description
            condition_folds = []
            for fold, (train, test) in enumerate(folds, 1):
                preprocess, model, _ = make_condition(condition)
                started = time.perf_counter()
                train_X = preprocess.fit_transform(X.iloc[train], y.iloc[train])
                test_X = preprocess.transform(X.iloc[test])
                if condition.endswith("without_interest"):
                    tokens = preprocess.named_transformers_["profile"]["tfidf"].get_feature_names_out()
                    assert not any("fields_of_interest=" in token for token in tokens)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(train_X, y.iloc[train])
                probabilities, predictions = model.predict_proba(test_X), model.predict(test_X)
                row = {
                    "condition": condition, "fold": fold, "train_rows": len(train), "test_rows": len(test),
                    "encoded_features": train_X.shape[1],
                    "top_1_accuracy": float(accuracy_score(y.iloc[test], predictions)),
                    "top_3_accuracy": float(top_k_accuracy_score(y.iloc[test], probabilities, k=3, labels=model.classes_)),
                    "macro_f1": float(f1_score(y.iloc[test], predictions, average="macro")),
                    "seconds": time.perf_counter() - started,
                    "iterations": int(np.max(model.n_iter_)),
                    "warnings": [str(w.message) for w in caught],
                }
                condition_folds.append(row)
                fold_rows.append(row)
                print(f"{condition}: fold {fold}/5 top-1={row['top_1_accuracy']:.4f} top-3={row['top_3_accuracy']:.4f} macro-F1={row['macro_f1']:.4f} seconds={row['seconds']:.1f}", flush=True)
                report["conditions"][condition]["folds"] = condition_folds
                save()
            summary = {"condition": condition, "cv_folds": len(folds)}
            summary.update(summarize_folds(
                condition_folds, ("top_1_accuracy", "top_3_accuracy", "macro_f1"),
            ))
            summary_rows.append(summary)
            report["conditions"][condition]["summary"] = summary
            save()
            print(f"Completed {condition}: {json.dumps(summary)}", flush=True)
    report["status"] = "complete"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
