@echo off
REM Enhanced NICEGOLD-ProjectP Pipeline Runner for Windows
REM ====================================================

echo.
echo ========================================
echo  NICEGOLD-ProjectP Full Pipeline
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Display current directory
echo Current directory: %CD%
echo.

REM Check if requirements are installed
echo Checking dependencies...
python -c "import pandas, numpy, sklearn, catboost; print('Dependencies OK')" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Missing dependencies detected. Installing...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Starting Enhanced Pipeline Runner...
echo.

REM Run the enhanced pipeline
python run_full_pipeline.py %*

if errorlevel 1 (
    echo.
    echo ========================================
    echo  Pipeline execution FAILED
    echo ========================================
    echo Check logs directory for details
) else (
    echo.
    echo ========================================
    echo  Pipeline execution COMPLETED
    echo ========================================
    echo Check output directory for results
)

echo.
pause
