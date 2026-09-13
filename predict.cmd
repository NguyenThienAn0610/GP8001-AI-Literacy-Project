@echo off
setlocal
pushd "%~dp0"

set "MODEL_NAME=%~1"
if not defined MODEL_NAME set "MODEL_NAME=logistic_regression"

set "INPUT_PATH=%~2"
if not defined INPUT_PATH set "INPUT_PATH=data\example_profiles.csv"

set "OUTPUT_PATH=%~3"
if not defined OUTPUT_PATH set "OUTPUT_PATH=predictions\quick_recommendations.csv"

set "MODEL_PATH=%MODEL_NAME%"
if not exist "%MODEL_PATH%" set "MODEL_PATH=models\%MODEL_NAME%.joblib"

if not exist "%MODEL_PATH%" (
    echo ERROR: Model not found: %MODEL_PATH%
    echo Train it first with: .\.venv\Scripts\python.exe src\train_models.py --model %MODEL_NAME%
    popd
    exit /b 1
)

if not exist "%INPUT_PATH%" (
    echo ERROR: Input CSV not found: %INPUT_PATH%
    popd
    exit /b 1
)

if defined CAREER_PYTHON (
    "%CAREER_PYTHON%" src\predict_career.py --model "%MODEL_PATH%" --input "%INPUT_PATH%" --output "%OUTPUT_PATH%" --top-k 3
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" src\predict_career.py --model "%MODEL_PATH%" --input "%INPUT_PATH%" --output "%OUTPUT_PATH%" --top-k 3
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.12 src\predict_career.py --model "%MODEL_PATH%" --input "%INPUT_PATH%" --output "%OUTPUT_PATH%" --top-k 3
    ) else (
        python src\predict_career.py --model "%MODEL_PATH%" --input "%INPUT_PATH%" --output "%OUTPUT_PATH%" --top-k 3
    )
)

if errorlevel 1 (
    popd
    exit /b 1
)

echo.
echo Top-3 recommendations:
set "QUICK_PREDICTION_OUTPUT=%OUTPUT_PATH%"
powershell -NoProfile -Command "Import-Csv -LiteralPath $env:QUICK_PREDICTION_OUTPUT | Select-Object Recommendation_1, Recommendation_1_Score, Recommendation_2, Recommendation_2_Score, Recommendation_3, Recommendation_3_Score | Format-Table -AutoSize"
echo Saved to: %OUTPUT_PATH%

popd
endlocal
