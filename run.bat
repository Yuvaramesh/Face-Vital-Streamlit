@echo off
REM Face Vital Monitor - Quick Launch Script for Windows

echo 🫀 Face Vital Monitor - Quick Launcher
echo ======================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed.
    pause
    exit /b 1
)

REM Install dependencies if requirements.txt exists
if exist requirements.txt (
    echo 📦 Installing/updating dependencies...
    pip install -r requirements.txt
)

REM Launch the Python launcher
python run_app.py
pause
