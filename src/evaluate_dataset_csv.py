"""Evaluate data/raw/dataset.csv as a job-sector recommendation dataset.

The target is Job_Sector. Only rows with a known sector can supervise the model.
By default, post-outcome columns (Employment_Status and Salary) are excluded to
avoid using information that is unavailable when making a career recommendation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Job_Sector"
POST_OUTCOME_COLUMNS = ["Employment_Status", "Salary"]


def plot_model_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot out-of-fold top-k scores beside their uniform-random baselines."""
    metrics = ["accuracy", "top_2_accuracy", "top_3_accuracy"]
    baselines = ["random_top_1", "random_top_2", "random_top_3"]
    labels = ["Top-1", "Top-2", "Top-3"]
    positions = np.arange(len(labels))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, row in enumerate(comparison.itertuples(index=False)):
        values = [getattr(row, metric) for metric in metrics]
        ax.bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            label=row.model,
        )
    chance = [comparison[column].iloc[0] for column in baselines]
    ax.plot(positions, chance, "o--", color="#111827", linewidth=2, label="Uniform random")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Out-of-fold accuracy")
    ax.set_ylim(0, 0.42)
    ax.set_title("dataset.csv model performance is at chance level")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(exclude=np.number).columns.tolist()
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def make_models(frame: pd.DataFrame, seed: int) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", make_preprocessor(frame)),
                (
                    "model",
                    LogisticRegression(max_iter=600, solver="lbfgs", random_state=seed),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("preprocess", make_preprocessor(frame)),
                (
                    "model",
                    ExtraTreesClassifier(
                        # Bound the trees: this dataset has 156k labeled rows and
                        # an unrestricted forest is needlessly expensive for a
                        # diagnostic comparison against chance.
                        n_estimators=80,
                        max_depth=16,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> tuple[dict[str, float], pd.DataFrame]:
    labels = np.sort(y.unique())
    probabilities = np.zeros((len(y), len(labels)), dtype=np.float32)
    predictions = np.empty(len(y), dtype=object)
    folds = np.zeros(len(y), dtype=np.int16)

    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        print(f"  fold {fold}/{cv.n_splits}", flush=True)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        fold_prob = model.predict_proba(x.iloc[test_idx])
        class_positions = {label: idx for idx, label in enumerate(labels)}
        for source_idx, label in enumerate(model.classes_):
            probabilities[test_idx, class_positions[label]] = fold_prob[:, source_idx]
        predictions[test_idx] = labels[np.argmax(probabilities[test_idx], axis=1)]
        folds[test_idx] = fold

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "top_2_accuracy": top_k_accuracy_score(y, probabilities, k=2, labels=labels),
        "top_3_accuracy": top_k_accuracy_score(y, probabilities, k=3, labels=labels),
        "log_loss": log_loss(y, probabilities, labels=labels),
    }
    oof = pd.DataFrame(
        {
            "row_index": y.index,
            "fold": folds,
            "actual": y.to_numpy(),
            "predicted": predictions,
            "confidence": probabilities.max(axis=1),
        }
    )
    for rank in range(3):
        ranked = np.argsort(-probabilities, axis=1)[:, rank]
        oof[f"recommendation_{rank + 1}"] = labels[ranked]
        oof[f"recommendation_{rank + 1}_probability"] = probabilities[
            np.arange(len(y)), ranked
        ]
    return metrics, oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/dataset.csv"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/dataset_csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/dataset_csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data)
    labeled = raw.loc[raw[TARGET].notna()].copy()
    y = labeled[TARGET].astype(str)
    feature_columns = [
        column
        for column in raw.columns
        if column not in [TARGET, *POST_OUTCOME_COLUMNS]
    ]
    x = labeled[feature_columns]
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    class_distribution = y.value_counts().rename_axis(TARGET).reset_index(name="count")
    class_distribution["proportion"] = class_distribution["count"] / len(y)
    class_distribution.to_csv(args.report_dir / "class_distribution.csv", index=False)

    results: list[dict[str, object]] = []
    for name, model in make_models(x, args.seed).items():
        print(f"Evaluating {name} with {args.folds}-fold out-of-fold predictions...", flush=True)
        metrics, oof = evaluate_model(model, x, y, cv)
        safe_name = name.lower().replace(" ", "_")
        oof.to_csv(args.report_dir / f"{safe_name}_oof_predictions.csv", index=False)
        results.append({"model": name, "cv_folds": args.folds, **metrics})
        model.fit(x, y)
        joblib.dump(model, args.model_dir / f"{safe_name}.joblib")

    comparison = pd.DataFrame(results)
    n_classes = y.nunique()
    comparison["random_top_1"] = 1 / n_classes
    comparison["random_top_2"] = 2 / n_classes
    comparison["random_top_3"] = 3 / n_classes
    comparison["accuracy_lift_over_random"] = comparison["accuracy"] / (1 / n_classes)
    comparison.to_csv(args.report_dir / "model_comparison.csv", index=False)
    plot_model_comparison(comparison, args.report_dir / "model_performance_vs_chance.png")

    metadata = {
        "source": str(args.data),
        "raw_rows": len(raw),
        "labeled_rows": len(labeled),
        "unlabeled_rows": int(raw[TARGET].isna().sum()),
        "target": TARGET,
        "classes": sorted(y.unique()),
        "feature_columns": feature_columns,
        "excluded_post_outcome_columns": POST_OUTCOME_COLUMNS,
        "cross_validation_folds": args.folds,
        "random_seed": args.seed,
    }
    (args.report_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
