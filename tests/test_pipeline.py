"""Regression checks for input handling, feature isolation, and scoring."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_feature_ablation import make_condition
from career_ml import FEATURE_COLUMNS, TARGET_COLUMN, ProfileTextTransformer, load_training_data
from evaluation import metric_values, summarize_folds, validation_key
from predict_career import add_ranked_recommendations


class PipelineTests(unittest.TestCase):
    def test_tokens_preserve_field_identity_and_missing_values(self):
        frame = pd.DataFrame({"Technical_Skills": [" Python, C++ "], "Achievements": [None]})
        self.assertEqual(ProfileTextTransformer().transform(frame).tolist(), [
            "technical_skills=python technical_skills=c++ achievements=missing",
        ])

    def test_training_data_rejects_missing_columns_and_labels(self):
        frame = pd.DataFrame({column: ["example"] for column in FEATURE_COLUMNS})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "must contain"):
                load_training_data(path)
            frame[TARGET_COLUMN] = None
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing labels"):
                load_training_data(path)
            frame.drop(columns="Age").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Age"):
                load_training_data(path)

    def test_ablation_excludes_interest_before_vocabulary_fitting(self):
        frame = pd.DataFrame({column: ["training", "training"] for column in FEATURE_COLUMNS})
        frame["Age"] = [20, 30]
        frame["Grade_CGPA_Percentage"] = [70, 80]
        frame["Fields_of_Interest"] = ["Secret label clue", "Another clue"]
        preprocessor, _, _ = make_condition("logistic_regression_without_interest")
        preprocessor.fit(frame)
        vocabulary = preprocessor.named_transformers_["profile"]["tfidf"].vocabulary_
        self.assertFalse(any("fields_of_interest=" in term for term in vocabulary))
        heldout = frame.copy()
        heldout["Technical_Skills"] = "unseenholdouttoken"
        preprocessor.transform(heldout)
        self.assertFalse(any("unseenholdouttoken" in term for term in vocabulary))

    def test_interest_only_selects_exactly_one_raw_field(self):
        preprocessor, _, _ = make_condition("logistic_regression_interest_only")
        selected = [column for _, _, columns in preprocessor.transformers for column in columns]
        self.assertEqual(selected, ["Fields_of_Interest"])

    def test_tied_probabilities_follow_sklearn_top_k_convention(self):
        labels = np.arange(5)
        probabilities = np.full((5, 5), 0.2)
        scores = metric_values(labels, probabilities.argmax(axis=1), probabilities, labels)
        self.assertAlmostEqual(scores["accuracy"], 0.2)
        self.assertAlmostEqual(scores["macro_f1"], 1 / 15)
        for k in (2, 3, 4):
            self.assertAlmostEqual(scores[f"top_{k}_accuracy"], k / 5)

    def test_summary_uses_population_standard_deviation(self):
        summary = summarize_folds([{"accuracy": 0.1}, {"accuracy": 0.3}], ["accuracy"], "cv_")
        self.assertAlmostEqual(summary["cv_accuracy_mean"], 0.2)
        self.assertAlmostEqual(summary["cv_accuracy_std"], 0.1)

    def test_validation_selection_prioritizes_accuracy_then_f1_then_loss(self):
        values = [
            {"accuracy": 0.3, "macro_f1": 0.9, "log_loss": 0.1},
            {"accuracy": 0.4, "macro_f1": 0.3, "log_loss": 0.2},
            {"accuracy": 0.4, "macro_f1": 0.4, "log_loss": 0.5},
            {"accuracy": 0.4, "macro_f1": 0.4, "log_loss": 0.4},
        ]
        self.assertIs(max(values, key=validation_key), values[-1])

    def test_recommendations_keep_labels_and_scores_aligned(self):
        class Model:
            classes_ = np.array(["A", "B", "C"])

            def predict_proba(self, features):
                return np.array([[0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])

        frame = pd.DataFrame({"profile": ["one", "two"]})
        result = add_ranked_recommendations(Model(), frame, frame.copy(), 2)
        self.assertEqual(result["Recommendation_1"].tolist(), ["B", "C"])
        np.testing.assert_array_equal(result["Recommendation_2_Score"], [0.2, 0.3])
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            add_ranked_recommendations(Model(), frame, frame.copy(), 4)


if __name__ == "__main__":
    unittest.main()
