#!/bin/bash
# ProjectP Quick Setup Script
# This script will install all required packages for ProjectP

echo "🚀 ProjectP - Quick Setup"
echo "========================="
echo ""

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    echo "💡 Please make sure you're in the ProjectP directory"
    exit 1
fi

echo "📦 Installing Python packages from requirements.txt..."
echo "⏱️  This may take a few minutes..."
echo ""

# Install packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "✅ Installation completed!"
echo ""
echo "🎯 You can now run ProjectP with:"
echo "   python ProjectP.py"
echo ""
echo "📊 Or use the module directly:"
echo "   python -m projectp --mode full_pipeline"
echo ""
echo "🌐 Or start the dashboard:"
echo "   streamlit run projectp/dashboard.py"
echo ""
echo "Good luck with your trading! 🚀"
