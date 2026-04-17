@echo off
title Project Slipstream - F1 Prediction Engine
color 0A

echo ============================================================
echo   🏎️  Project Slipstream — Full Setup, Train ^& Launch
echo ============================================================
echo.

:: ──────────────────────────────────────────────
:: Step 0 — Resolve project root (where this .bat lives)
:: ──────────────────────────────────────────────
pushd "%~dp0"
set "PROJECT_ROOT=%cd%"
echo [INFO] Project root: %PROJECT_ROOT%
echo.

:: ──────────────────────────────────────────────
:: Step 1 — Check Python is available
:: ──────────────────────────────────────────────
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not on PATH.
    echo         Please install Python 3.10+ and try again.
    pause
    exit /b 1
)
python --version
echo       Python found. ✓
echo.

:: ──────────────────────────────────────────────
:: Step 2 — Create virtual environment (if needed)
:: ──────────────────────────────────────────────
echo [2/6] Setting up virtual environment...
if not exist "%PROJECT_ROOT%\venv" (
    echo       Creating virtual environment...
    python -m venv "%PROJECT_ROOT%\venv"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Virtual environment created. ✓
) else (
    echo       Virtual environment already exists. ✓
)

:: Activate the virtual environment
call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
echo       Virtual environment activated. ✓
echo.

:: ──────────────────────────────────────────────
:: Step 3 — Install dependencies
:: ──────────────────────────────────────────────
echo [3/6] Installing Python dependencies...
pip install -r "%PROJECT_ROOT%\requirements.txt" --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo       All dependencies installed. ✓
echo.

:: ──────────────────────────────────────────────
:: Step 4 — Run Data Pipeline (fetch + cache F1 data)
:: ──────────────────────────────────────────────
echo [4/6] Running data pipeline (fetching F1 data)...
echo       This may take a while on first run due to API calls.
echo.
python "%PROJECT_ROOT%\src\data_pipeline.py"
if %errorlevel% neq 0 (
    echo [ERROR] Data pipeline failed.
    pause
    exit /b 1
)
echo       Data pipeline complete. ✓
echo.

:: ──────────────────────────────────────────────
:: Step 5 — Feature Engineering
:: ──────────────────────────────────────────────
echo [5/6] Running feature engineering...
python "%PROJECT_ROOT%\src\feature_engineering.py"
if %errorlevel% neq 0 (
    echo [ERROR] Feature engineering failed.
    pause
    exit /b 1
)
echo       Feature engineering complete. ✓
echo.

:: ──────────────────────────────────────────────
:: Step 6 — Model Training
:: ──────────────────────────────────────────────
echo [6/6] Training ML model (GridSearchCV + evaluation)...
echo       This may take a few minutes...
echo.
python "%PROJECT_ROOT%\src\model_training.py"
if %errorlevel% neq 0 (
    echo [ERROR] Model training failed.
    pause
    exit /b 1
)
echo       Model training complete. ✓
echo.

:: ──────────────────────────────────────────────
:: Launch — Streamlit Dashboard
:: ──────────────────────────────────────────────
echo ============================================================
echo   ✅  Setup ^& Training Complete!
echo   🚀  Launching Streamlit Dashboard...
echo ============================================================
echo.
echo   The dashboard will open in your default browser.
echo   Press Ctrl+C in this window to stop the server.
echo.

python -m streamlit run "%PROJECT_ROOT%\src\app.py"

:: Cleanup
popd
pause
