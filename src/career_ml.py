"""Shared preprocessing and model definitions for career-path prediction."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Suggested_Career_Path"

NUMERIC_COLUMNS = ["Age", "Grade_CGPA_Percentage"]

PROFILE_COLUMNS = [
    "Gender",
    "Location",
    "Highest_qualification",
    "Stream",
    "Current_Academic_Level",
    "Technical_Skills",
    "Soft_Skills",
    "Languages_Known",
    "Certifications",
    "Fields_of_Interest",
    "Preferred_Work_Style",
    "Work_Type_Interest",
    "Past_Jobs_Internships",
    "Achievements",
    "Skills_Gained",
    "Willing_to_Relocate",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + PROFILE_COLUMNS


class ProfileTextTransformer(BaseEstimator, TransformerMixin):
    """Convert mixed profile fields to field-aware tokens.

    Comma-separated values such as skills and languages become separate tokens.
    Prefixing each token with its source column prevents, for example, a skill
    value from being confused with a qualification value.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ProfileTextTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(X, columns=getattr(X, "columns", PROFILE_COLUMNS))
        documents: list[str] = []

        for values in frame.itertuples(index=False, name=None):
            tokens: list[str] = []
            for column, value in zip(frame.columns, values):
                raw_value = "missing" if pd.isna(value) else str(value)
                for item in raw_value.split(","):
                    normalized = re.sub(
                        r"[^a-z0-9+#]+", "_", item.strip().lower()
                    ).strip("_")
                    if normalized:
                        tokens.append(f"{column.lower()}={normalized}")
            documents.append(" ".join(tokens))

        return np.asarray(documents)


def make_preprocessor() -> ColumnTransformer:
    """Build preprocessing that can be serialized with each trained model."""
    profile_pipeline = Pipeline(
        steps=[
            ("profile_tokens", ProfileTextTransformer()),
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=str.split,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("profile", profile_pipeline, PROFILE_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ]
    )


MODEL_FACTORIES: dict[str, Callable[[], BaseEstimator]] = {
    "extra_trees": lambda: ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    ),
    "logistic_regression": lambda: LogisticRegression(
        C=16.0,
        max_iter=3_000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    ),
}

# Five-fold validation showed that binary presence/absence features generalize
# better for the tree ensembles. Logistic Regression remains stronger with the
# default TF-IDF weighting and bigrams.
MODEL_PREPROCESSOR_PARAMS: dict[str, dict[str, object]] = {
    "extra_trees": {
        "profile__tfidf__ngram_range": (1, 1),
        "profile__tfidf__use_idf": False,
        "profile__tfidf__binary": True,
        "profile__tfidf__norm": None,
    },
    "random_forest": {
        "profile__tfidf__ngram_range": (1, 1),
        "profile__tfidf__use_idf": False,
        "profile__tfidf__binary": True,
        "profile__tfidf__norm": None,
    },
    "logistic_regression": {},
}


def make_pipeline(model_name: str) -> Pipeline:
    """Create one of the three benchmarked end-to-end classifiers."""
    if model_name not in MODEL_FACTORIES:
        valid = ", ".join(MODEL_FACTORIES)
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {valid}")

    preprocessor = make_preprocessor().set_params(
        **MODEL_PREPROCESSOR_PARAMS[model_name]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", MODEL_FACTORIES[model_name]()),
        ]
    )


def validate_features(frame: pd.DataFrame) -> None:
    """Fail early with a useful message when input columns are missing."""
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")
