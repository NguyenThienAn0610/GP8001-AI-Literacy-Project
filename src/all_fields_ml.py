"""Preprocessing and model definitions for the all-fields career dataset."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Career"
CATEGORICAL_COLUMNS = ["Field"]
NUMERIC_COLUMNS = [
    "GPA",
    "Extracurricular_Activities",
    "Internships",
    "Projects",
    "Leadership_Positions",
    "Field_Specific_Courses",
    "Research_Experience",
    "Coding_Skills",
    "Communication_Skills",
    "Problem_Solving_Skills",
    "Teamwork_Skills",
    "Analytical_Skills",
    "Presentation_Skills",
    "Networking_Skills",
    "Industry_Certifications",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


MODEL_FACTORIES: dict[str, Callable[[], BaseEstimator]] = {
    "logistic_regression": lambda: LogisticRegression(
        C=1.0,
        max_iter=3_000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    ),
    "extra_trees": lambda: ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
}


def validate_features(columns: list[str]) -> None:
    missing = sorted(set(FEATURE_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")


def make_pipeline(model_name: str) -> Pipeline:
    if model_name not in MODEL_FACTORIES:
        valid = ", ".join(MODEL_FACTORIES)
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {valid}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", MODEL_FACTORIES[model_name]()),
        ]
    )
