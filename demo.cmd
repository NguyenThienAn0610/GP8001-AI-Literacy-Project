@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Create the environment first: python -m venv .venv
    echo Then install packages: .venv\Scripts\python.exe -m pip install -r requirements.txt
    popd
    exit /b 1
)
".venv\Scripts\python.exe" src\demo_app.py %*
set "DEMO_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %DEMO_EXIT_CODE%
