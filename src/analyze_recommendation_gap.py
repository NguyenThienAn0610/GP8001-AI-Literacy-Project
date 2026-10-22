"""Reproduce the interest-only baseline and describe the Top-1/Top-3 gap.

Training-fold frequencies alone determine each held-out row's ranked list.
Full-dataset counts below are descriptive; they are never used to predict a
held-out row. This diagnostic does not establish a theoretical accuracy ceiling
or validate suitability of the recommended careers in the real world.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


INTEREST = "Fields_of_Interest"
TARGET = "Suggested_Career_Path"


def ranked_labels(counts: pd.Series) -> list[str]:
    """Descending training frequency; alphabetical label order breaks ties."""
    return sorted(counts.index.tolist(), key=lambda label: (-int(counts[label]), label))


def score_predictions(truth: np.ndarray, ranked: list[list[str]]) -> dict[str, float]:
    first = np.asarray([labels[0] for labels in ranked])
    return {
        "top1_accuracy": float(accuracy_score(truth, first)),
        "macro_f1": float(f1_score(truth, first, average="macro", zero_division=0)),
        "top3_accuracy": float(np.mean([label in choices[:3] for label, choices in zip(truth, ranked)])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/Career_Path_Dataset_with_Career.csv"))
    parser.add_argument("--report-prefix", type=Path, default=Path("reports/recommendation_gap"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    if data[[INTEREST, TARGET]].isna().any().any():
        raise ValueError("This diagnostic requires nonmissing interest and target labels.")
    interests = data[INTEREST].astype(str)
    targets = data[TARGET].astype(str)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows: list[dict] = []
    predictions: list[dict] = []
    oof_ranked: list[list[str] | None] = [None] * len(data)

    for fold, (train_idx, test_idx) in enumerate(cv.split(interests, targets), start=1):
        train = data.iloc[train_idx]
        lookup = {
            str(interest): ranked_labels(group[TARGET].astype(str).value_counts())
            for interest, group in train.groupby(INTEREST)
        }
        fallback = ranked_labels(train[TARGET].astype(str).value_counts())
        ranked = [lookup.get(interest, fallback) for interest in interests.iloc[test_idx]]
        fold_metrics = score_predictions(targets.iloc[test_idx].to_numpy(), ranked)
        rows.append({"fold": fold, "train_rows": len(train_idx), "test_rows": len(test_idx), **fold_metrics})
        for index, choices in zip(test_idx, ranked):
            oof_ranked[int(index)] = choices
            predictions.append({
                "row_index": int(index),
                "fold": fold,
                "interest": interests.iloc[index],
                "actual_career": targets.iloc[index],
                "rank_1": choices[0],
                "rank_2": choices[1] if len(choices) > 1 else "",
                "rank_3": choices[2] if len(choices) > 2 else "",
                "top1_correct": targets.iloc[index] == choices[0],
                "top3_correct": targets.iloc[index] in choices[:3],
            })

    if any(choices is None for choices in oof_ranked):
        raise AssertionError("Every row must receive exactly one held-out prediction.")
    fold_frame = pd.DataFrame(rows)
    prediction_frame = pd.DataFrame(predictions).sort_values("row_index")
    if prediction_frame["row_index"].nunique() != len(data):
        raise AssertionError("Held-out row indices must not repeat.")
    cross_tab = pd.crosstab(interests, targets)
    long_counts = cross_tab.stack().rename("count").reset_index()
    long_counts = long_counts.loc[long_counts["count"] > 0].copy()
    long_counts["interest_rows"] = long_counts[INTEREST].map(interests.value_counts())
    long_counts["within_interest_percentage"] = long_counts["count"] / long_counts["interest_rows"] * 100
    interest_counts = interests.value_counts().sort_index()
    distinct_careers = (cross_tab > 0).sum(axis=1)
    career_counts = targets.value_counts().sort_index()
    software_rows = int(interest_counts.get("Software Development", 0))
    software_share = software_rows / len(data)
    metrics = ["top1_accuracy", "macro_f1", "top3_accuracy"]
    output = {
        "method": {
            "command": "python src/analyze_recommendation_gap.py --folds 5 --seed 42",
            "data_path": args.data.as_posix(),
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "folds": args.folds,
            "stratified_by": TARGET,
            "shuffle": True,
            "random_state": args.seed,
            "baseline": "Rank careers by frequency within Fields_of_Interest using training-fold rows only.",
            "unseen_interest_fallback": "Global career frequencies from the same training fold.",
            "ties": "Alphabetical ascending career label after descending training frequency.",
            "macro_f1": "Unweighted mean across all 22 observed classes; undefined per-class F1 is zero.",
            "standard_deviation": "Population standard deviation across folds (ddof=0), matching train_models.py.",
        },
        "dataset": {
            "rows": len(data),
            "input_features": len(data.columns) - 1,
            "career_classes": int(targets.nunique()),
            "interest_groups": int(interests.nunique()),
            "class_counts": {str(key): int(value) for key, value in career_counts.items()},
            "smallest_class_count": int(career_counts.min()),
            "largest_class_count": int(career_counts.max()),
            "largest_to_smallest_class_ratio": float(career_counts.max() / career_counts.min()),
            "uniform_random_22_class_top1_reference": float(1 / targets.nunique()),
            "uniform_random_22_class_top3_reference": float(3 / targets.nunique()),
            "interest_counts": {str(key): int(value) for key, value in interest_counts.items()},
            "careers_per_interest": {str(key): int(value) for key, value in distinct_careers.items()},
            "every_career_occurs_in_one_interest_group": bool(((cross_tab > 0).sum(axis=0) == 1).all()),
            "software_development_rows": software_rows,
            "software_development_share": software_share,
            "data_science_distribution": long_counts.loc[long_counts[INTEREST] == "Data Science"].to_dict(orient="records"),
        },
        "interest_only_cross_validation": {
            "fold_metrics": rows,
            "mean": {metric: float(fold_frame[metric].mean()) for metric in metrics},
            "std": {metric: float(fold_frame[metric].std(ddof=0)) for metric in metrics},
            "out_of_fold_metrics": score_predictions(targets.to_numpy(), oof_ranked),
            "distinct_top1_predictions_across_all_folds": int(prediction_frame["rank_1"].nunique()),
            "top1_correct_rows": int(prediction_frame["top1_correct"].sum()),
            "top3_correct_rows": int(prediction_frame["top3_correct"].sum()),
            "all_top3_errors_in_software_development": bool((prediction_frame.loc[~prediction_frame["top3_correct"], "interest"] == "Software Development").all()),
        },
        "uniform_within_interest_illustration": {
            "assumptions": "Interest group is identified correctly; careers within each group are equally likely and cannot otherwise be distinguished. This is an illustrative approximation, not an empirically proven upper bound.",
            "weighted_top1": float(sum(interest_counts[name] / len(data) / distinct_careers[name] for name in interest_counts.index)),
            "weighted_top3": float(sum(interest_counts[name] / len(data) * min(3 / distinct_careers[name], 1) for name in interest_counts.index)),
            "top3_formula": "(1 - Software Development share) * 1 + Software Development share * 3/4",
        },
        "interpretation_limits": [
            "Top-1 requires the saved label to be ranked first; Top-3 only requires it to appear anywhere in a three-item shortlist.",
            "Six interest groups contain three careers; one contains four. A model can identify the career family yet struggle to rank within it.",
            "Nearly uniform labels within each interest group alone do not prove that remaining features contain no useful signal.",
            "Comparison against full-feature models supports examining limited within-group signal, but this diagnostic does not establish a Bayes accuracy ceiling.",
            "Results measure reproduction of dataset labels under stratified cross-validation, not career suitability or real-world recommendation success.",
        ],
    }
    args.report_prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(args.report_prefix)
    Path(prefix + ".json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    cross_tab.to_csv(prefix + "_interest_career_counts.csv")
    long_counts.to_csv(prefix + "_interest_career_distribution.csv", index=False)
    fold_frame.to_csv(prefix + "_fold_metrics.csv", index=False)
    prediction_frame.to_csv(prefix + "_oof_predictions.csv", index=False)
    pd.DataFrame({"career": career_counts.index, "count": career_counts.values, "percentage": career_counts.values / len(data) * 100}).to_csv(prefix + "_class_balance.csv", index=False)
    print(json.dumps({
        "mean_cv_metrics": output["interest_only_cross_validation"]["mean"],
        "std_cv_metrics": output["interest_only_cross_validation"]["std"],
        "data_science_distribution": output["dataset"]["data_science_distribution"],
        "software_development_share": software_share,
        "illustrative_top1": output["uniform_within_interest_illustration"]["weighted_top1"],
        "illustrative_top3": output["uniform_within_interest_illustration"]["weighted_top3"],
        "report_prefix": prefix,
    }, indent=2))


if __name__ == "__main__":
    main()
