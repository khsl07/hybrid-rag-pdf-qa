@echo off
echo ========================================================
echo   Hybrid RAG PDF Q^&A System - Setup and Launch
echo ========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist "venv" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
)

echo [2/3] Activating virtual environment and installing dependencies...
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo [3/3] Launching Streamlit application...
echo App will open in your default browser at http://localhost:8501
streamlit run app.py
pause
