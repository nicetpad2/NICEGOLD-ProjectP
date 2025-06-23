@echo off
echo ========================================
echo COMPLETE PROJECT SETUP AND RUN
echo ========================================
cd /d "g:\My Drive\Phiradon1688_co"

echo.
echo [1/3] Training required models...
python complete_model_training.py
if errorlevel 1 (
    echo ERROR: Model training failed
    pause
    exit /b 1
)

echo.
echo [2/3] Running emergency AUC fixes...
python emergency_nan_auc_fix.py
if errorlevel 1 (
    echo WARNING: Emergency fix had issues, continuing...
)

echo.
echo [3/3] Running main project pipeline...
python ProjectP.py --run_full_pipeline
if errorlevel 1 (
    echo ERROR: Main pipeline failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo PROJECT COMPLETED SUCCESSFULLY!
echo Check output_default/ for results
echo ========================================
pause
