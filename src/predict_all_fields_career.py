"""Generate ranked predictions from a trained all-fields career model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from all_fields_ml import FEATURE_COLUMNS, TARGET_COLUMN, validate_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions/all_fields_career_predictions.csv"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input, skipinitialspace=True)
    data.columns = data.columns.str.strip()
    validate_features(data.columns.tolist())

    model = joblib.load(args.model)
    classes = np.asarray(model.classes_)
    if args.top_k < 1 or args.top_k > len(classes):
        raise ValueError(f"--top-k must be between 1 and {len(classes)}.")

    probabilities = np.asarray(model.predict_proba(data[FEATURE_COLUMNS]))
    ranked = np.argsort(probabilities, axis=1)[:, ::-1][:, : args.top_k]
    output = data.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    if TARGET_COLUMN in data.columns:
        output[f"Actual_{TARGET_COLUMN}"] = data[TARGET_COLUMN]
    for rank in range(args.top_k):
        indices = ranked[:, rank]
        output[f"Recommendation_{rank + 1}"] = classes[indices]
        output[f"Recommendation_{rank + 1}_Score"] = probabilities[
            np.arange(len(output)), indices
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote predictions for {len(output):,} rows to {args.output}")


if __name__ == "__main__":
    main()
