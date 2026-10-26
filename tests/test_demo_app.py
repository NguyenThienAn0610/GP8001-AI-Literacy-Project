"""Exercise the demo's real-model predictions and local HTTP boundary."""

import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from career_ml import FEATURE_COLUMNS
from demo_app import CareerDemo, DEFAULT_MODEL, MAX_BODY_BYTES, create_server
from predict_career import add_ranked_recommendations


@unittest.skipUnless(DEFAULT_MODEL.exists(), "Train Logistic Regression before running demo integration tests.")
class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = CareerDemo()
        cls.assets = tempfile.TemporaryDirectory()
        Path(cls.assets.name, "index.html").write_text("<!doctype html><title>Demo</title>")
        cls.server = create_server(cls.service, port=0, static_dir=cls.assets.name)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.assets.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_real_model_scores_match_existing_cli_function(self):
        profile = self.service.config["example"]
        frame = pd.DataFrame([profile], columns=FEATURE_COLUMNS)
        expected = add_ranked_recommendations(self.service.model, frame, frame.copy(), 3).iloc[0]
        actual = self.service.predict(profile)
        for result in actual["recommendations"]:
            rank = result["rank"]
            self.assertEqual(result["career"], expected[f"Recommendation_{rank}"])
            self.assertEqual(result["score"], expected[f"Recommendation_{rank}_Score"])

    def test_blank_optional_fields_still_produce_finite_scores(self):
        profile = {name: None for name in FEATURE_COLUMNS}
        profile["Fields_of_Interest"] = "Data Science"
        result = self.service.predict(profile)
        self.assertEqual(len(result["recommendations"]), 3)
        self.assertTrue(all(np.isfinite(row["score"]) for row in result["recommendations"]))
        self.assertIsNone(result["profile"]["Age"])

    def test_invalid_numeric_categories_and_schema_are_rejected(self):
        profile = self.service.config["example"]
        for field, value in [("Age", True), ("Age", float("inf")), ("Age", 10 ** 500), ("Age", 24.5),
                             ("Grade_CGPA_Percentage", "NaN"), ("Grade_CGPA_Percentage", 101),
                             ("Fields_of_Interest", ""), ("Fields_of_Interest", "Not a category"),
                             ("Location", ["not text"]), ("Location", "x" * 1001)]:
            with self.subTest(field=field, value=str(value)[:30]), self.assertRaises(ValueError):
                self.service.predict({**profile, field: value})
        with self.assertRaisesRegex(ValueError, "Missing profile fields"):
            self.service.predict({"Fields_of_Interest": "Data Science"})
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.service.predict({**profile, "Suggested_Career_Path": "Data Scientist"})

    def test_http_prediction_and_form_configuration(self):
        status, headers, body = self.request("GET", "/api/config")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        config = json.loads(body)
        self.assertEqual(len(config["fields"]), 18)
        self.assertEqual(config["career_count"], 22)
        status, _, body = self.request("POST", "/api/predict", json.dumps({"profile": config["example"]}),
                                       {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["recommendations"]), 3)

    def test_static_routes_cannot_expose_project_files(self):
        status, _, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>Demo</title>", body)
        for path in ["/../models/logistic_regression.joblib", "/%2e%2e/data/example_profiles.csv",
                     "/data/raw/Career_Path_Dataset_with_Career.csv", "/api/unknown"]:
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_http_rejects_bad_json_large_body_and_cross_origin(self):
        cases = [
            ("not json", {"Content-Type": "application/json"}, 400),
            ("{}", {"Content-Type": "text/plain"}, 415),
            ("x" * (MAX_BODY_BYTES + 1), {"Content-Type": "application/json"}, 413),
            ("{}", {"Content-Type": "application/json", "Origin": "https://unrelated.example"}, 403),
        ]
        for body, headers, expected in cases:
            with self.subTest(status=expected):
                status, _, response = self.request("POST", "/api/predict", body, headers)
                self.assertEqual(status, expected)
                self.assertIn("error", json.loads(response))


if __name__ == "__main__":
    unittest.main()
