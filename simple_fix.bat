@echo off
cd /d "g:\My Drive\Phiradon1688_co"
echo Running Emergency NaN AUC Fix Scripts...
echo.

echo Step 1: Quick Diagnosis
python quick_nan_auc_diagnosis.py
echo.

echo Step 2: Emergency Fix  
python emergency_nan_auc_fix.py
echo.

echo Step 3: Critical Fix
python critical_nan_auc_fix.py
echo.

echo All fixes completed!
echo Check output_default folder for results
pause
