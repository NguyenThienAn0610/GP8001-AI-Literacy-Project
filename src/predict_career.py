"""Return ranked career recommendations for new profile rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from career_ml import FEATURE_COLUMNS, TARGET_COLUMN, validate_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trained .joblib model.")
    parser.add_argument("--input", type=Path, required=True, help="CSV of new profiles.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions/career_predictions.csv"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of ranked career recommendations to return (default: 3).",
    )
    return parser.parse_args()


def add_ranked_recommendations(
    model: object,
    features: pd.DataFrame,
    output: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Add the top-k career names and probability scores to an output frame."""
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if not hasattr(model, "predict_proba"):
        raise TypeError("The selected model does not support probability rankings.")
    if not hasattr(model, "classes_"):
        raise TypeError("The selected model does not expose trained class labels.")

    classes = np.asarray(model.classes_)
    if top_k > len(classes):
        raise ValueError(
            f"--top-k cannot exceed the model's {len(classes)} career classes."
        )

    probabilities = np.asarray(model.predict_proba(features))
    ranked_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :top_k]

    for rank in range(top_k):
        indices = ranked_indices[:, rank]
        output[f"Recommendation_{rank + 1}"] = classes[indices]
        output[f"Recommendation_{rank + 1}_Score"] = probabilities[
            np.arange(len(output)), indices
        ]

    return output


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    validate_features(frame)

    model = joblib.load(args.model)
    features = frame[FEATURE_COLUMNS]
    output = frame.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    output = add_ranked_recommendations(model, features, output, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(
        f"Wrote {args.top_k} ranked recommendations for "
        f"{len(output):,} profile(s) to {args.output}"
    )


if __name__ == "__main__":
    main()
