@echo off
echo ==========================================
echo NaN AUC EMERGENCY FIX - Critical Issue
echo ==========================================
cd /d "g:\My Drive\Phiradon1688_co"

echo.
echo [1/4] Running quick NaN AUC diagnosis...
python quick_nan_auc_diagnosis.py
if errorlevel 1 (
    echo WARNING: Quick diagnosis had issues, continuing...
)

echo.
echo [2/4] Running emergency NaN AUC fix...
python emergency_nan_auc_fix.py
if errorlevel 1 (
    echo WARNING: Emergency fix had issues, continuing...
)

echo.
echo [3/4] Running critical NaN AUC fix...
python critical_nan_auc_fix.py
if errorlevel 1 (
    echo WARNING: Critical fix had issues, continuing...
)

echo.
echo [4/4] Running original critical imbalance fix...
python critical_auc_fix.py
if errorlevel 1 (
    echo WARNING: Original fix had issues, continuing...
)

echo.
echo ==========================================
echo ALL EMERGENCY FIXES COMPLETED
echo Check output_default folder for results
echo ==========================================

echo.
echo Results Summary:
echo Looking for generated files...
dir output_default\*.txt 2>nul && (
    echo [FOUND] TXT reports generated
) || (
    echo [MISSING] No TXT reports found
)

dir output_default\*.json 2>nul && (
    echo [FOUND] JSON reports generated
) || (
    echo [MISSING] No JSON reports found
)

dir output_default\*.csv 2>nul && (
    echo [FOUND] CSV data files generated
) || (
    echo [MISSING] No CSV data files found
)

echo.
echo Next steps:
echo 1. Check output_default/ folder for detailed reports
echo 2. Look for AUC scores in the output above
echo 3. If still getting NaN, check package installations
echo 4. Run: pip install scikit-learn pandas numpy

pause
