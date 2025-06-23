@echo off
REM ==========================================
REM 🚀 INSTANT AUC FIX BATCH SCRIPT
REM ==========================================
REM Windows batch script to fix AUC issues instantly

echo.
echo ==========================================
echo 🚀 INSTANT AUC FIX LAUNCHER
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python detected
echo.

REM Show menu
echo Select fix type:
echo 1. Quick Fix (Recommended)
echo 2. Emergency Hotfix Only
echo 3. Complete Production Fix
echo 4. Check Status Only
echo 5. Start Monitoring
echo.

set /p choice="Enter choice (1-5): "

echo.
echo 🔧 Running AUC fix...
echo.

if "%choice%"=="1" (
    echo 🚀 Running Quick Fix...
    python fix_auc_now.py
) else if "%choice%"=="2" (
    echo 🚨 Running Emergency Hotfix...
    python fix_auc_now.py --emergency
) else if "%choice%"=="3" (
    echo 🔧 Running Complete Fix...
    python fix_auc_now.py --full
) else if "%choice%"=="4" (
    echo 🔍 Checking Status...
    python fix_auc_now.py --status
) else if "%choice%"=="5" (
    echo 🎯 Starting Monitor...
    python fix_auc_now.py --monitor
) else (
    echo ❌ Invalid choice. Running default quick fix...
    python fix_auc_now.py
)

echo.
echo ==========================================
echo 🏁 AUC Fix Process Completed
echo ==========================================
echo.

REM Optional: Run the main pipeline to test
set /p runpipeline="Do you want to test the pipeline now? (y/n): "
if /i "%runpipeline%"=="y" (
    echo.
    echo 🧪 Testing pipeline...
    python ProjectP.py --run_full_pipeline
)

echo.
echo 📋 Summary:
echo ✅ AUC fix process completed
echo 📁 Check output_default/ folder for results
echo 🔍 Review logs for detailed information
echo.

pause
