#!/bin/bash
# -*- coding: utf-8 -*-
"""
🚀 NICEGOLD ProjectP - Enhanced Full Pipeline Launcher
Quick start script for running enhanced full pipeline with advanced results summary
"""

echo "🚀 NICEGOLD ProjectP - Enhanced Full Pipeline"
echo "============================================="
echo ""
echo "🎯 Starting Enhanced Full Pipeline with Advanced Results Summary..."
echo ""

# Set the working directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    echo "💡 Please install Python 3.8+ to continue"
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "✅ Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "✅ Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️ No virtual environment found, using system Python"
fi

# Run the enhanced pipeline test
echo "🔄 Running Enhanced Full Pipeline Test..."
echo ""

python3 test_direct_pipeline.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉 ENHANCED FULL PIPELINE COMPLETED SUCCESSFULLY!"
    echo "✅ Advanced Results Summary has been generated"
    echo "📁 Check the results/summary/ directory for detailed reports"
    echo ""
    echo "🚀 Your NICEGOLD ProjectP system is ready for production!"
else
    echo ""
    echo "❌ Enhanced Full Pipeline failed with exit code: $exit_code"
    echo "💡 Please check the error messages above and try again"
    exit $exit_code
fi
