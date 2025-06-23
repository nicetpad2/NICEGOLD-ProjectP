@echo off
echo ==========================================
echo NaN AUC EMERGENCY FIX - Critical Issue
echo ==========================================
cd /d "g:\My Drive\Phiradon1688_co"

echo.
echo [1/3] Running quick NaN AUC diagnosis...
python quick_nan_auc_diagnosis.py

echo.
echo [2/3] Running emergency NaN AUC fix...  
python emergency_nan_auc_fix.py

echo.
echo [3/3] Running critical imbalance fix...
python critical_auc_fix.py

echo.
echo ==========================================
echo EMERGENCY FIXES COMPLETED
echo Check output_default folder for results
echo ==========================================

echo.
echo Results Summary:
dir output_default\*.txt 2>nul && echo - TXT reports found || echo - No TXT reports
dir output_default\*.json 2>nul && echo - JSON reports found || echo - No JSON reports  
dir output_default\*.csv 2>nul && echo - CSV data found || echo - No CSV data

pause
