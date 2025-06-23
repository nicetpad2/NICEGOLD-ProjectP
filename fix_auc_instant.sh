#!/bin/bash

# ==========================================
# 🚀 INSTANT AUC FIX SCRIPT
# ==========================================
# Linux/Mac bash script to fix AUC issues instantly

echo ""
echo "=========================================="
echo "🚀 INSTANT AUC FIX LAUNCHER"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please install Python first."
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "✅ Python detected ($PYTHON_CMD)"
echo ""

# Show menu
echo "Select fix type:"
echo "1. Quick Fix (Recommended)"
echo "2. Emergency Hotfix Only"
echo "3. Complete Production Fix"
echo "4. Check Status Only"
echo "5. Start Monitoring"
echo "6. Run Full Pipeline with AUC Fix"
echo "7. Install Requirements & Quick Fix"
echo ""

read -p "Enter choice (1-7): " choice

echo ""

# Install requirements if choice 7 or if requirements.txt exists
if [ "$choice" = "7" ] || [ -f "requirements.txt" ]; then
    echo "📦 Installing/updating requirements..."
    $PYTHON_CMD -m pip install -r requirements.txt --user --quiet 2>/dev/null || echo "⚠️ Some packages may need manual installation"
    echo "✅ Requirements processing completed"
    echo ""
fi

echo "🔧 Running AUC fix..."
echo ""

case $choice in
    1)
        echo "🚀 Running Quick Fix..."
        $PYTHON_CMD fix_auc_now.py
        ;;
    2)
        echo "🚨 Running Emergency Hotfix..."
        $PYTHON_CMD fix_auc_now.py --emergency
        ;;
    3)
        echo "🔧 Running Complete Fix..."
        $PYTHON_CMD fix_auc_now.py --full
        ;;
    4)
        echo "🔍 Checking Status..."
        $PYTHON_CMD fix_auc_now.py --status
        ;;
    5)
        echo "🎯 Starting Monitor..."
        $PYTHON_CMD fix_auc_now.py --monitor
        ;;
    6)
        echo "🚀 Running Full Pipeline with Integrated AUC Fix..."
        $PYTHON_CMD ProjectP.py --run_full_pipeline
        ;;
    7)
        echo "🚀 Running Quick Fix after requirements install..."
        $PYTHON_CMD fix_auc_now.py
        ;;
    *)
        echo "❌ Invalid choice. Running default quick fix..."
        $PYTHON_CMD fix_auc_now.py
        ;;
esac
        echo "🚨 Running Emergency Hotfix..."
        $PYTHON_CMD fix_auc_now.py --emergency
        ;;
    3)
        echo "🔧 Running Complete Fix..."
        $PYTHON_CMD fix_auc_now.py --full
        ;;
    4)
        echo "🔍 Checking Status..."
        $PYTHON_CMD fix_auc_now.py --status
        ;;
    5)
        echo "🎯 Starting Monitor..."
        $PYTHON_CMD fix_auc_now.py --monitor
        ;;
    *)
        echo "❌ Invalid choice. Running default quick fix..."
        $PYTHON_CMD fix_auc_now.py
        ;;
esac

echo ""
echo "=========================================="
echo "🏁 AUC Fix Process Completed"
echo "=========================================="
echo ""

# Optional: Run the main pipeline to test
read -p "Do you want to test the pipeline now? (y/n): " runpipeline
if [[ $runpipeline == "y" || $runpipeline == "Y" ]]; then
    echo ""
    echo "🧪 Testing pipeline..."
    $PYTHON_CMD ProjectP.py --run_full_pipeline
fi

echo ""
echo "📋 Summary:"
echo "✅ AUC fix process completed"
echo "📁 Check output_default/ folder for results"
echo "🔍 Review logs for detailed information"
echo ""
