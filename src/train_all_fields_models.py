"""Train and evaluate career models on career_path_in_all_field.csv."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from all_fields_ml import (
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
        default=Path("data/raw/career_path_in_all_field.csv"),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", choices=list(MODEL_FACTORIES))
    selection.add_argument("--models", nargs="+", choices=list(MODEL_FACTORIES))
    selection.add_argument("--all-models", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model-dir", type=Path, default=Path("models/all_fields"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/all_fields"))
    return parser.parse_args()


def selected_models(args: argparse.Namespace) -> list[str]:
    if args.model:
        return [args.model]
    if args.models:
        return list(dict.fromkeys(args.models))
    return list(MODEL_FACTORIES)


def top_k_accuracy(
    true_labels: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> float:
    ranked = np.argsort(probabilities, axis=1)[:, -k:]
    true_indices = pd.Categorical(true_labels, categories=classes).codes
    return float(np.mean((ranked == true_indices[:, None]).any(axis=1)))


def make_prediction_table(
    data: pd.DataFrame,
    true_labels: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    ranked = np.argsort(probabilities, axis=1)[:, ::-1][:, :top_k]
    output = pd.DataFrame(
        {
            "Source_Row": np.arange(2, len(data) + 2),
            "Actual_Career": true_labels.to_numpy(),
            "Predicted_Career": classes[ranked[:, 0]],
        }
    )
    output["Correct"] = output["Actual_Career"] == output["Predicted_Career"]
    for rank in range(top_k):
        indices = ranked[:, rank]
        output[f"Recommendation_{rank + 1}"] = classes[indices]
        output[f"Recommendation_{rank + 1}_Score"] = probabilities[
            np.arange(len(output)), indices
        ]
    return output


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("Top-1", "cv_accuracy_mean", "chance_top_1"),
        ("Top-3", "cv_top_3_accuracy_mean", "chance_top_3"),
        ("Top-5", "cv_top_5_accuracy_mean", "chance_top_5"),
    ]
    labels = [item[0] for item in metrics]
    x = np.arange(len(metrics), dtype=float)
    series_count = len(comparison) + 1
    width = 0.8 / series_count
    colors = ["#2563EB", "#7C3AED", "#0F766E"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_index, row in enumerate(comparison.itertuples(index=False)):
        values = [getattr(row, item[1]) for item in metrics]
        positions = x - 0.4 + width / 2 + model_index * width
        bars = ax.bar(
            positions,
            values,
            width,
            label=row.model.replace("_", " ").title(),
            color=colors[model_index % len(colors)],
        )
        ax.bar_label(bars, labels=[f"{value:.2%}" for value in values], fontsize=8)

    chance_values = [comparison[item[2]].iloc[0] for item in metrics]
    chance_positions = x - 0.4 + width / 2 + len(comparison) * width
    chance_bars = ax.bar(
        chance_positions,
        chance_values,
        width,
        label="Random chance",
        color="#9CA3AF",
    )
    ax.bar_label(
        chance_bars,
        labels=[f"{value:.2%}" for value in chance_values],
        fontsize=8,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_title("All-fields dataset performance versus random chance")
    ax.legend(frameon=False, ncols=2)
    ax.set_ylim(0, max(chance_values) * 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")

    data = pd.read_csv(args.data, skipinitialspace=True)
    data.columns = data.columns.str.strip()
    validate_features(data.columns.tolist())
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET_COLUMN}'.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains missing values.")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].astype(str)
    classes = np.sort(y.unique())
    if args.top_k < 1 or args.top_k > len(classes):
        raise ValueError(f"--top-k must be between 1 and {len(classes)}.")
    if y.value_counts().min() < args.cv_folds:
        raise ValueError("Every target class must have at least cv-folds rows.")

    folds = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for model_name in selected_models(args):
        print(f"Evaluating {model_name} ...", flush=True)
        base_pipeline = make_pipeline(model_name)
        predictions = pd.Series(index=y.index, dtype="object")
        probabilities = np.zeros((len(y), len(classes)), dtype=float)
        fold_accuracy: list[float] = []
        fold_macro_f1: list[float] = []
        fold_weighted_f1: list[float] = []
        fold_top_2: list[float] = []
        fold_top_3: list[float] = []
        fold_top_5: list[float] = []
        started = time.perf_counter()

        for fold_number, (train_indices, test_indices) in enumerate(
            folds.split(X, y), start=1
        ):
            pipeline = clone(base_pipeline)
            pipeline.fit(X.iloc[train_indices], y.iloc[train_indices])
            fold_predictions = pipeline.predict(X.iloc[test_indices])
            fold_probabilities = pipeline.predict_proba(X.iloc[test_indices])
            if not np.array_equal(pipeline.classes_, classes):
                raise RuntimeError("Fold class ordering differed from the global classes.")

            probabilities[test_indices] = fold_probabilities
            predictions.iloc[test_indices] = fold_predictions
            fold_y = y.iloc[test_indices]
            fold_accuracy.append(accuracy_score(fold_y, fold_predictions))
            fold_macro_f1.append(
                f1_score(fold_y, fold_predictions, average="macro")
            )
            fold_weighted_f1.append(
                f1_score(fold_y, fold_predictions, average="weighted")
            )
            fold_top_2.append(top_k_accuracy(fold_y, fold_probabilities, classes, 2))
            fold_top_3.append(top_k_accuracy(fold_y, fold_probabilities, classes, 3))
            fold_top_5.append(top_k_accuracy(fold_y, fold_probabilities, classes, 5))
            print(f"  completed fold {fold_number}/{args.cv_folds}", flush=True)

        row: dict[str, object] = {
            "model": model_name,
            "rows": len(data),
            "features": len(FEATURE_COLUMNS),
            "classes": len(classes),
            "cv_folds": args.cv_folds,
            "cv_accuracy_mean": np.mean(fold_accuracy),
            "cv_accuracy_std": np.std(fold_accuracy),
            "cv_macro_f1_mean": np.mean(fold_macro_f1),
            "cv_macro_f1_std": np.std(fold_macro_f1),
            "cv_weighted_f1_mean": np.mean(fold_weighted_f1),
            "cv_weighted_f1_std": np.std(fold_weighted_f1),
            "cv_top_2_accuracy_mean": np.mean(fold_top_2),
            "cv_top_3_accuracy_mean": np.mean(fold_top_3),
            "cv_top_5_accuracy_mean": np.mean(fold_top_5),
            "chance_top_1": 1 / len(classes),
            "chance_top_2": 2 / len(classes),
            "chance_top_3": 3 / len(classes),
            "chance_top_5": 5 / len(classes),
            "cv_seconds": time.perf_counter() - started,
        }

        report = classification_report(y, predictions, output_dict=True, zero_division=0)
        report_path = args.report_dir / f"{model_name}_classification_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        prediction_table = make_prediction_table(
            data, y, probabilities, classes, args.top_k
        )
        prediction_table.to_csv(
            args.report_dir / f"{model_name}_out_of_fold_predictions.csv",
            index=False,
        )

        print(f"Fitting final {model_name} model on all rows ...", flush=True)
        final_pipeline = make_pipeline(model_name)
        fit_started = time.perf_counter()
        final_pipeline.fit(X, y)
        row["full_fit_seconds"] = time.perf_counter() - fit_started
        joblib.dump(
            final_pipeline,
            args.model_dir / f"{model_name}.joblib",
            compress=3,
        )
        results.append(row)
        print(
            f"  accuracy={row['cv_accuracy_mean']:.4f}, "
            f"macro-F1={row['cv_macro_f1_mean']:.4f}, "
            f"top-3={row['cv_top_3_accuracy_mean']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(results).sort_values(
        "cv_macro_f1_mean", ascending=False
    )
    comparison.to_csv(args.report_dir / "model_comparison.csv", index=False)
    plot_model_comparison(
        comparison, args.report_dir / "model_performance_vs_chance.png"
    )
    print("\nModel comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
