@echo off
echo ==========================================
echo NaN AUC EMERGENCY FIX - Critical Issue
echo ==========================================
cd /d "g:\My Drive\Phiradon1688_co"
echo.
echo � Running simple validation test...
python simple_auc_test.py
echo.
echo � Running critical imbalance fix...
python critical_auc_fix.py
echo.
echo ==========================================
echo 🎯 TESTS COMPLETED - Check Results Above
echo ==========================================
pause
pause
