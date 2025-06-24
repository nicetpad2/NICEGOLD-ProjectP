@echo off
REM ProjectP Quick Setup Script for Windows
REM This script will install all required packages for ProjectP

echo 🚀 ProjectP - Quick Setup
echo =========================
echo.

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ❌ requirements.txt not found!
    echo 💡 Please make sure you're in the ProjectP directory
    pause
    exit /b 1
)

echo 📦 Installing Python packages from requirements.txt...
echo ⏱️  This may take a few minutes...
echo.

REM Install packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ✅ Installation completed!
echo.
echo 🎯 You can now run ProjectP with:
echo    python ProjectP.py
echo.
echo 📊 Or use the module directly:
echo    python -m projectp --mode full_pipeline
echo.
echo 🌐 Or start the dashboard:
echo    streamlit run projectp/dashboard.py
echo.
echo Good luck with your trading! 🚀
pause
