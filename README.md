# GP8001 AI Literacy Project

This project trains multiclass classifiers to predict `Suggested_Career_Path`
from education, skills, interests, experience, and work-preference data.

## Dataset findings

- 15,000 rows and 18 input features
- 22 near-balanced career classes
- Missing values occur in `Certifications`, `Past_Jobs_Internships`, and
  `Achievements`; the pipeline converts them to an explicit `missing` token.
- Comma-separated skills, languages, and certifications are split into
  field-aware tokens before binary or TF-IDF encoding.

The selected models were compared with stratified five-fold cross-validation
(`random_state=42`). Macro-F1 is the primary metric because it gives equal
weight to every career class.

| Rank | Model | CV Accuracy | CV Macro-F1 | Why include it |
|---:|---|---:|---:|---|
| 1 | Logistic Regression | 0.3282 | 0.3248 | Best tuned result; `C=16` preserves weak secondary signals. |
| 2 | Random Forest | 0.3249 | 0.3195 | Binary unigram features reduce sensitivity to noisy combinations. |
| 3 | Extra Trees | 0.3225 | 0.3173 | Binary unigram features improve both accuracy and macro-F1. |

Ranked recommendation performance:

| Model | Top-1 | Top-2 | Top-3 | Top-4 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.3282 | 0.6483 | **0.9649** | 0.9987 |
| Random Forest | 0.3249 | 0.6434 | 0.9634 | 1.0000 |
| Extra Trees | 0.3225 | 0.6411 | 0.9618 | 1.0000 |

### Why accuracy stops near 33%

The raw data makes `Fields_of_Interest` the only strong predictor. Each interest
maps to three possible careers, except `Software Development`, which maps to
four. Within each interest group, the career labels are distributed almost
uniformly and the remaining features are close to random. A five-fold lookup
using only the most common career per interest reaches 0.3299 accuracy but only
0.1578 macro-F1 because it never predicts most careers.

The tuned models are therefore close to the honest accuracy ceiling while still
predicting all 22 career classes. Predictions should be treated as suggestions,
not definitive career advice. Meaningfully higher accuracy requires data in
which skills, qualifications, experience, or preferences actually distinguish
careers within the same interest group.

### Tuning changes

- Extra Trees: TF-IDF bigrams replaced with binary unigrams.
- Random Forest: TF-IDF bigrams replaced with binary unigrams.
- Logistic Regression: regularization parameter `C` increased from 4 to 16.
- Evaluation increased from three-fold to five-fold stratified validation.

## Install

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Required Python packages:

- `pandas` — CSV loading and output
- `numpy` — numeric operations
- `scikit-learn` — preprocessing, models, splitting, and metrics
- `joblib` — saving and loading trained pipelines

## Train models

Train all three models; both commands are equivalent:

```powershell
python src/train_models.py
python src/train_models.py --all-models
```

This writes serialized pipelines to `models/` and evaluation files to
`reports/`. Each saved pipeline contains its preprocessing, which prevents
training/prediction feature mismatches.

Train one specific model:

```powershell
python src/train_models.py --model logistic_regression
python src/train_models.py --model random_forest
python src/train_models.py --model extra_trees
```

Train a selected group of models:

```powershell
python src/train_models.py --models logistic_regression random_forest
```

The `--model`, `--models`, and `--all-models` options are mutually exclusive.

## Generate top-3 career recommendations

Quickest option from the project root:

```powershell
.\predict.cmd
```

This uses Logistic Regression with `data\example_profiles.csv`, prints the
top-three recommendations, and saves them to
`predictions\quick_recommendations.csv`.

Optionally provide a model name, input CSV, and output CSV:

```powershell
.\predict.cmd random_forest data\my_profiles.csv predictions\my_results.csv
```

Prepare a CSV with the same 18 input columns as `data/example_profiles.csv`.
It must not require a `Suggested_Career_Path` value.

```powershell
python src/predict_career.py `
  --model models/logistic_regression.joblib `
  --input data/example_profiles.csv `
  --output predictions/career_predictions.csv
```

The output contains three ranked career names and scores:

- `Recommendation_1` and `Recommendation_1_Score`
- `Recommendation_2` and `Recommendation_2_Score`
- `Recommendation_3` and `Recommendation_3_Score`

Use `--top-k` to request a different number of recommendations:

```powershell
python src/predict_career.py `
  --model models/logistic_regression.joblib `
  --input data/example_profiles.csv `
  --top-k 3
```

The training report includes cross-validated top-2, top-3, and top-4 accuracy
for every model. Logistic Regression is recommended for ranked output because
it produced the strongest five-fold top-1 and macro-F1 results.
