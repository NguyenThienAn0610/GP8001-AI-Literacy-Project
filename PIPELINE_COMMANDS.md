# Career Recommendation Pipeline Commands

Run these commands in Windows PowerShell from the project directory.

## 1. Open the project

```powershell
Set-Location E:\Code\GP8001-AI-Literacy-Project
```

## 2. Create the Python environment

```powershell
py -3.12 -m venv .venv
```

If the `py` launcher is unavailable:

```powershell
python -m venv .venv
```

## 3. Install dependencies

These commands use the environment's Python directly, so activation is not
required.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Train models

### Train all models

Omitting a selection trains all models:

```powershell
.\.venv\Scripts\python.exe src\train_models.py
```

The explicit equivalent is:

```powershell
.\.venv\Scripts\python.exe src\train_models.py --all-models
```

### Train one model

Recommended model:

```powershell
.\.venv\Scripts\python.exe src\train_models.py --model logistic_regression
```

Other individual choices:

```powershell
.\.venv\Scripts\python.exe src\train_models.py --model random_forest
.\.venv\Scripts\python.exe src\train_models.py --model extra_trees
```

### Train multiple selected models

List the model names after `--models`:

```powershell
.\.venv\Scripts\python.exe src\train_models.py --models logistic_regression random_forest
```

Another valid combination:

```powershell
.\.venv\Scripts\python.exe src\train_models.py --models random_forest extra_trees
```

Available model names are:

```text
logistic_regression
random_forest
extra_trees
```

Training uses five-fold stratified cross-validation by default. To change the
number of folds:

```powershell
.\.venv\Scripts\python.exe src\train_models.py --model logistic_regression --cv-folds 3
```

## 5. Review evaluation results

```powershell
Import-Csv reports\model_comparison.csv |
    Select-Object model, cv_accuracy_mean, cv_top_2_accuracy_mean,
                  cv_top_3_accuracy_mean, cv_top_4_accuracy_mean |
    Format-Table -AutoSize
```

Generated files include:

```text
models\logistic_regression.joblib
models\random_forest.joblib
models\extra_trees.joblib
reports\model_comparison.csv
reports\*_classification_report.json
```

When only one model is trained, `model_comparison.csv` contains that run only.
Existing model files for models not selected are left unchanged.

## 6. Test the supplied example profile

The shortest command is:

```powershell
.\predict.cmd
```

It defaults to:

```text
Model:  models\logistic_regression.joblib
Input:  data\example_profiles.csv
Output: predictions\quick_recommendations.csv
Top-k:  3
```

To override the model, input, and output:

```powershell
.\predict.cmd random_forest data\my_profiles.csv predictions\my_results.csv
```

The full Python command remains available:

```powershell
.\.venv\Scripts\python.exe src\predict_career.py `
    --model models\logistic_regression.joblib `
    --input data\example_profiles.csv `
    --output predictions\career_recommendations.csv `
    --top-k 3
```

## 7. View the top-3 recommendations

```powershell
Import-Csv predictions\career_recommendations.csv |
    Select-Object Recommendation_1, Recommendation_1_Score,
                  Recommendation_2, Recommendation_2_Score,
                  Recommendation_3, Recommendation_3_Score |
    Format-Table -AutoSize
```

## 8. Test your own profiles

Copy the example input and edit its values while keeping the same column names:

```powershell
Copy-Item data\example_profiles.csv data\my_profiles.csv
notepad data\my_profiles.csv
```

Generate recommendations:

```powershell
.\.venv\Scripts\python.exe src\predict_career.py `
    --model models\logistic_regression.joblib `
    --input data\my_profiles.csv `
    --output predictions\my_recommendations.csv `
    --top-k 3
```

## 9. Compare prediction models

Random Forest:

```powershell
.\.venv\Scripts\python.exe src\predict_career.py `
    --model models\random_forest.joblib `
    --input data\my_profiles.csv `
    --output predictions\random_forest_recommendations.csv `
    --top-k 3
```

Extra Trees:

```powershell
.\.venv\Scripts\python.exe src\predict_career.py `
    --model models\extra_trees.joblib `
    --input data\my_profiles.csv `
    --output predictions\extra_trees_recommendations.csv `
    --top-k 3
```

## Quick start

For a complete run using the recommended model:

```powershell
Set-Location E:\Code\GP8001-AI-Literacy-Project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\train_models.py --model logistic_regression
.\.venv\Scripts\python.exe src\predict_career.py --model models\logistic_regression.joblib --input data\example_profiles.csv --output predictions\career_recommendations.csv --top-k 3
Import-Csv predictions\career_recommendations.csv | Format-Table -AutoSize
```
