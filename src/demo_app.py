"""Serve the local career recommendation demo using the saved model.

Run from any directory: python /path/to/project/src/demo_app.py
This development server is intended for local demonstrations.
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd

from career_ml import FEATURE_COLUMNS, NUMERIC_COLUMNS
from predict_career import add_ranked_recommendations

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/logistic_regression.joblib"
MAX_BODY_BYTES = 32 * 1024
SELECT_FIELDS = {
    "Gender", "Highest_qualification", "Stream", "Current_Academic_Level",
    "Fields_of_Interest", "Preferred_Work_Style", "Work_Type_Interest", "Willing_to_Relocate",
}
LABELS = {
    "Age": "Age", "Grade_CGPA_Percentage": "Grade / percentage",
    "Gender": "Gender", "Location": "Location", "Highest_qualification": "Highest qualification",
    "Stream": "Field of study", "Current_Academic_Level": "Current stage",
    "Technical_Skills": "Technical skills", "Soft_Skills": "Soft skills",
    "Languages_Known": "Languages", "Certifications": "Certifications",
    "Fields_of_Interest": "Career interest", "Preferred_Work_Style": "Work arrangement",
    "Work_Type_Interest": "Type of work", "Past_Jobs_Internships": "Work experience",
    "Achievements": "Achievements", "Skills_Gained": "Skills gained",
    "Willing_to_Relocate": "Willing to relocate",
}
PLACEHOLDERS = {
    "Location": "e.g. Singapore", "Technical_Skills": "e.g. Python, SQL, Data Analysis",
    "Soft_Skills": "e.g. Communication, Teamwork", "Languages_Known": "e.g. English",
    "Certifications": "e.g. Python, Data Science", "Past_Jobs_Internships": "e.g. Internship at XYZ",
    "Achievements": "e.g. Top Performer", "Skills_Gained": "e.g. Programming",
}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class CareerDemo:
    """Load the model and form choices once; score profiles without saving them."""

    def __init__(self, model_path: Path = DEFAULT_MODEL):
        self.model = joblib.load(model_path)
        if not hasattr(self.model, "predict_proba") or not hasattr(self.model, "classes_"):
            raise ValueError("Choose a fitted model that supports probability rankings.")
        estimator = self.model.named_steps.get("model") if hasattr(self.model, "named_steps") else self.model
        self.model_name = {
            "LogisticRegression": "Logistic Regression",
            "RandomForestClassifier": "Random Forest", "ExtraTreesClassifier": "Extra Trees",
        }.get(type(estimator).__name__, type(estimator).__name__)
        data = pd.read_csv(ROOT / "data/raw/Career_Path_Dataset_with_Career.csv")
        self.options = {name: sorted(data[name].dropna().astype(str).unique().tolist())
                        for name in SELECT_FIELDS}
        fields = []
        for name in FEATURE_COLUMNS:
            field = {"name": name, "label": LABELS[name], "required": name == "Fields_of_Interest"}
            if name in NUMERIC_COLUMNS:
                field.update(type="number", min=0, max=120 if name == "Age" else 100,
                             step=1 if name == "Age" else 0.1,
                             observed_min=float(data[name].min()), observed_max=float(data[name].max()))
            elif name in SELECT_FIELDS:
                field.update(type="select", options=self.options[name])
            else:
                field.update(type="text", placeholder=PLACEHOLDERS[name])
            fields.append(field)
        example = pd.read_csv(ROOT / "data/example_profiles.csv", skipinitialspace=True)
        example.columns = example.columns.str.strip()
        profile = {name: None if pd.isna(value) else value.strip() if isinstance(value, str)
                   else value.item() if isinstance(value, np.generic) else value
                   for name, value in example.iloc[0][FEATURE_COLUMNS].items()}
        self.config = {"fields": fields, "example": self.normalize(profile),
                       "model": self.model_name, "career_count": len(self.model.classes_)}

    def normalize(self, profile: dict) -> dict:
        if not isinstance(profile, dict):
            raise ValueError("Profile must be a JSON object.")
        missing = set(FEATURE_COLUMNS) - profile.keys()
        if missing:
            raise ValueError("Missing profile fields: " + ", ".join(sorted(missing)))
        if profile.keys() - set(FEATURE_COLUMNS):
            raise ValueError("Profile contains unsupported fields.")
        clean = {}
        for name in FEATURE_COLUMNS:
            value = profile[name]
            if isinstance(value, str):
                value = value.strip()
            if value is None or value == "":
                if name == "Fields_of_Interest":
                    raise ValueError("Choose a career interest.")
                clean[name] = None
                continue
            if name in NUMERIC_COLUMNS:
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    raise ValueError(f"{LABELS[name]} must be a number.")
                try:
                    value = float(value)
                except (ValueError, OverflowError):
                    raise ValueError(f"{LABELS[name]} must be a number.") from None
                maximum = 120 if name == "Age" else 100
                if not math.isfinite(value) or not 0 <= value <= maximum:
                    raise ValueError(f"{LABELS[name]} must be between 0 and {maximum}.")
                if name == "Age" and not value.is_integer():
                    raise ValueError("Age must be a whole number.")
            elif not isinstance(value, str) or len(value) > 1000:
                raise ValueError(f"{LABELS[name]} must be text of at most 1,000 characters.")
            elif name in self.options and value not in self.options[name]:
                raise ValueError(f"Choose a listed option for {LABELS[name].lower()}.")
            clean[name] = value
        return clean

    def predict(self, profile: dict) -> dict:
        clean = self.normalize(profile)
        # Match CSV missing-value semantics for the serialized preprocessing.
        frame = pd.DataFrame([{key: np.nan if value is None else value for key, value in clean.items()}],
                             columns=FEATURE_COLUMNS)
        ranked = add_ranked_recommendations(self.model, frame, pd.DataFrame(index=frame.index), 3).iloc[0]
        return {
            "model": self.model_name, "profile": clean,
            "recommendations": [
                {"rank": rank, "career": str(ranked[f"Recommendation_{rank}"]),
                 "score": float(ranked[f"Recommendation_{rank}_Score"])} for rank in (1, 2, 3)
            ],
        }


def create_server(service: CareerDemo, host="127.0.0.1", port=8000, static_dir=ROOT / "demo"):
    """Return a server without starting it, allowing ephemeral-port integration tests."""
    static_dir = Path(static_dir)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            pass  # Keep submitted profiles and request paths out of logs.

        def respond(self, status, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/api/config":
                self.respond(200, service.config)
            elif route == "/api/health":
                self.respond(200, {"status": "ok", "model": service.model_name})
            elif route in STATIC_FILES:
                filename, content_type = STATIC_FILES[route]
                try:
                    self.respond(200, (static_dir / filename).read_bytes(), content_type)
                except FileNotFoundError:
                    self.respond(404, {"error": "Demo asset not found."})
            else:
                self.respond(404, {"error": "Route not found."})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/predict":
                self.respond(404, {"error": "Route not found."})
                return
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.respond(403, {"error": "Cross-origin requests are not allowed."})
                return
            if self.headers.get_content_type() != "application/json":
                self.respond(415, {"error": "Send the profile as application/json."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.respond(400, {"error": "Invalid request length."})
                return
            if not 0 < length <= MAX_BODY_BYTES:
                self.respond(413 if length > MAX_BODY_BYTES else 400, {"error": "Send a JSON body smaller than 32 KB."})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != {"profile"}:
                    raise ValueError("Send an object containing a profile field.")
                result = service.predict(payload["profile"])
            except (ValueError, UnicodeDecodeError) as exc:
                self.respond(400, {"error": str(exc) if not isinstance(exc, json.JSONDecodeError) else "The request is not valid JSON."})
                return
            except Exception:
                self.respond(500, {"error": "Prediction failed. Please try again or restart the demo."})
                return
            self.respond(200, result)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; defaults to this computer only.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535.")
    try:
        service = CareerDemo(args.model)
        server = create_server(service, args.host, args.port)
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot start the demo: {exc}. Train a model with src/train_models.py if needed.")
    print(f"Career demo: http://{args.host}:{server.server_port}", flush=True)
    print("Press Ctrl+C to stop. Submitted profiles are not saved.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
