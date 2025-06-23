@echo off
chcp 65001 >nul
title ULTIMATE PIPELINE WITH EMERGENCY FIXES

echo.
echo ====================================================================
echo 🔥 ULTIMATE PIPELINE WITH EMERGENCY FIXES INTEGRATION
echo ====================================================================
echo 🚀 This script runs the ULTIMATE pipeline with:
echo    ✨ Emergency AUC fixes
echo    🧠 Advanced feature engineering  
echo    🎯 Model ensemble improvements
echo    📊 Full pipeline integration
echo ====================================================================
echo.

echo 🔍 Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python.
    pause
    exit /b 1
)

echo ✅ Python found!
echo.

echo 🔧 Installing required packages...
pip install pandas numpy scikit-learn --quiet
echo ✅ Basic packages installed
echo.

echo 🚀 Starting ULTIMATE Pipeline...
echo ====================================================================
python ProjectP.py --ultimate_pipeline

if %errorlevel% equ 0 (
    echo.
    echo ====================================================================
    echo 🎉 ULTIMATE PIPELINE COMPLETED SUCCESSFULLY!
    echo ====================================================================
    echo 📁 Check output_default/ folder for results
    echo 📊 Check emergency_fixes.log for detailed fix logs
    echo.
    
    if exist "output_default\emergency_fixed_ultimate_pipeline.csv" (
        echo ✅ Emergency fixed data: output_default\emergency_fixed_ultimate_pipeline.csv
    )
    
    if exist "output_default\ultimate_results" (
        echo ✅ Ultimate results: output_default\ultimate_results
    )
    
    if exist "models\auc_improvement_config.json" (
        echo ✅ AUC improvement config: models\auc_improvement_config.json
    )
    
) else (
    echo.
    echo ====================================================================
    echo ❌ ULTIMATE PIPELINE FAILED
    echo ====================================================================
    echo 🔧 Please check the error messages above
    echo 📋 Check emergency_fixes.log for detailed logs
)

echo.
echo Press any key to exit...
pause >nul
