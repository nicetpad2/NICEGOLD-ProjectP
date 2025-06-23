@echo off
chcp 65001 >nul
cd /d "g:\My Drive\Phiradon1688_co"

echo ========================================
echo Emergency NaN AUC Fix
echo ========================================

echo [1/1] Running Emergency NaN AUC Fix...
python emergency_nan_auc_fix.py

echo.
echo ========================================
echo Emergency Fix Completed!
echo Check output_default folder for results
echo ========================================
pause
