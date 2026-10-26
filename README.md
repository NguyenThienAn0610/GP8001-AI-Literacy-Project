# GP8001 AI Literacy: Career Recommendations

This course project predicts `Suggested_Career_Path` from a person's education,
skills, interests, experience, and work preferences. It returns a ranked career
shortlist and compares traditional machine learning with deep learning models.

The dataset contains **15,000 profiles, 18 input fields, and 22 career labels**.
Results measure agreement with those labels, not real-world career suitability.

## Quick start

Run these commands in PowerShell from the project root. Use **Python 3.11 or
later**, as required by the pinned scikit-learn version. The existing environment
was tested with Python 3.14.7. PyTorch is optional and is only needed for the
deep learning benchmark.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src/train_models.py --model logistic_regression
.\predict.cmd
```

If `python` is unavailable, use the Windows `py` launcher to create the virtual
environment. The commands use its interpreter directly; activation is optional.
Skip training if `models/logistic_regression.joblib` already exists.

`predict.cmd` reads [the example profile](data/example_profiles.csv), prints
three recommendations, and writes `predictions/quick_recommendations.csv`.

## Try the web demo

After installing the base requirements and training Logistic Regression, start
the browser interface from the project root:

```powershell
.\demo.cmd
```

Open **http://127.0.0.1:8000**. Select **Try an example**, then **Explore my
careers** to get three recommendations from the saved model. You can edit the
profile, expand the education and experience sections, and download the results
as a CSV. Only the main interest is required; blank fields use the pipeline's
existing missing-value handling.

The demo uses the same prediction function as the command-line application.
There are no additional packages or frontend build steps. To use another port:

```powershell
.venv\Scripts\python.exe src/demo_app.py --port 8001
```

Keep the terminal running while presenting, and press **Ctrl+C** to stop.
Profiles are processed in memory and are not saved by the server. CSV downloads
are saved only when you request them in the browser.

The implementation is a small example of a real application flow:
**profile form → JSON API → saved preprocessing and model → ranked results**.
The interface is in `demo/`; `src/demo_app.py` serves it and exposes
`GET /api/config` and `POST /api/predict`. The prediction request has the shape
`{"profile": {"Age": 24, "Fields_of_Interest": "Data Science", ...}}`, with all
18 field names present and `null` or empty strings for optional blanks.

This is a local demonstration server. A public service would need production
hosting, HTTPS, appropriate access controls, and evaluation on representative
real-world profiles. The displayed scores describe the model's career labels,
not a person's chance of success.

## Predict for your own profiles

Create a CSV using the same 18 input column names as the example. Each row is one
profile; no target label is needed. Quote comma-separated skills or other lists
so they remain in one CSV cell. Missing text values become a `missing` token;
missing numeric values use medians learned during training.

```powershell
.\predict.cmd logistic_regression data/my_profiles.csv predictions/my_results.csv
```

For more control, call the Python script directly:

```powershell
.venv\Scripts\python.exe src/predict_career.py `
  --model models/logistic_regression.joblib `
  --input data/example_profiles.csv `
  --output predictions/career_predictions.csv `
  --top-k 3
```

The output retains the input fields and adds `Recommendation_1` through
`Recommendation_3`, each with a corresponding `_Score` column. Scores are model
probabilities, not validated probabilities of career success.

## Train and compare models

```powershell
# Train all three saved-model options.
.venv\Scripts\python.exe src/train_models.py

# Or select a subset.
.venv\Scripts\python.exe src/train_models.py --models logistic_regression random_forest
```

Available models are `logistic_regression`, `random_forest`, and `extra_trees`.
The script evaluates each using five stratified folds with seed 42, then fits
the final pipeline on all rows. Preprocessing is fitted separately inside each
fold and saved with the final model.

Outputs include `models/<model>.joblib`, per-model classification reports, and
`reports/model_comparison.csv`. A run replaces the selected model files and
writes a comparison containing only that run's models. Use `--model-dir` and
`--report-dir` to keep a separate run. Use `--help` on any script for its options.

## What the results show

These are mean scores across the same five outer folds. The neural models use
an inner validation split to select settings within each outer training fold.

| Model | Top-1 accuracy | Macro-F1 | Top-3 accuracy |
|---|---:|---:|---:|
| Logistic Regression | **32.82%** | **0.3248** | **96.49%** |
| Random Forest | 32.49% | 0.3195 | 96.34% |
| Extra Trees | 32.25% | 0.3173 | 96.18% |
| Tuned deep MLP | 32.32% | 0.2439 | 95.48% |
| Tuned field Transformer | 32.23% | 0.2353 | 96.47% |

Sources: [traditional model results](reports/model_comparison.csv) and
[deep learning results](reports/deep_learning_search/model_comparison.csv).
Logistic Regression remains the default because it achieved the strongest
Top-1 and macro-F1 among the evaluated approaches.

- **Top-1:** the first predicted career matches the recorded label.
- **Top-3:** the recorded label appears anywhere in the three recommendations.
- **Macro-F1:** average per-career F1, giving every career equal weight. F1
  balances precision and recall; a higher value is better.

The large Top-1/Top-3 gap comes mainly from the dataset's interest groups. Six
of the seven groups contain three career labels; Software Development contains
four. Models identify the group well, but the other features provide limited
evidence for choosing one career within it. High Top-3 coverage therefore does
not imply equally strong first-choice prediction.

The deep learning search tested 18 configurations over five outer folds, varying
architecture, learning rate, regularization, and scheduling. It did not improve
the best Top-1 or macro-F1 result. See [experiment details](docs/experiments.md)
for the full comparison, architecture specifications, feature relevance
analysis, limitations, and commands to reproduce or audit the results.

## Project layout

| Path | Purpose |
|---|---|
| `src/career_ml.py` | Shared input schema, preprocessing, and saved-model definitions |
| `src/train_models.py`, `src/predict_career.py` | Training and prediction commands |
| `demo/`, `src/demo_app.py`, `demo.cmd` | Browser demo, prediction API, and Windows launcher |
| `src/evaluation.py`, `src/experiment_models.py`, `src/neural_models.py` | Shared evaluation and experiment code |
| `src/benchmark_*.py`, `src/analyze_*.py`, `src/summarize_deep_learning.py` | Benchmarks, feature diagnostics, and result audit |
| `data/` | Raw dataset and example prediction input |
| `models/`, `predictions/`, `reports/` | Saved pipelines, recommendations, and evaluation evidence |
| `presentations/` | Project slides |
| `tests/` | Regression checks for preprocessing, evaluation, and feature selection |
| `archive/` | Original prototype retained for reference |

The current deck is [AI Application and Results](presentations/GP8001_GE2_AI_Application_Results.pptx).

Run the regression checks with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
