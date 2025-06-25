#!/bin/bash
# Enhanced Pipeline Launcher
# Run the NICEGOLD ProjectP Enhanced Full Pipeline with Professional Summary

echo "🚀 NICEGOLD ProjectP - Enhanced Full Pipeline Launcher"
echo "════════════════════════════════════════════════════════"
echo "🎯 Running enhanced full pipeline with professional trading summary..."
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Run the main ProjectP with enhanced pipeline
python3 ProjectP_refactored.py << EOF
1
EOF

echo ""
echo "✅ Enhanced pipeline completed!"
echo "🏆 Professional trading summary generated with:"
echo "   • Starting capital tracking"
echo "   • Win/Loss rates analysis"  
echo "   • Maximum Drawdown calculation"
echo "   • Complete test period analysis"
echo "   • Professional trading metrics"
echo "   • Executive summary format"
echo ""
echo "📁 Check results/ directory for detailed reports"
echo "🎯 Ready for next development phase!"
