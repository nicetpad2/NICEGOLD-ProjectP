#!/bin/bash
# 🚀 NICEGOLD ProjectP AI Agents Ultimate Launcher

echo "🚀 NICEGOLD ProjectP AI Agents Ultimate System"
echo "════════════════════════════════════════════════"
echo ""
echo "เลือกการเริ่มต้น:"
echo "1. 🌟 Ultimate Web Interface (แนะนำ - ระบบเทพ!)"
echo "2. 🎯 Enhanced Web Interface" 
echo "3. 🔧 Clean Web Interface"
echo "4. 📱 Basic Web Interface"
echo "5. 💻 CLI Simple Runner"
echo ""

read -p "เลือก (1-5): " choice

case $choice in
    1)
        echo "🚀 เริ่มระบบเว็บ Ultimate..."
        echo "URL: http://localhost:8503"
        streamlit run ai_agents_web_ultimate.py --server.port 8503
        ;;
    2)
        echo "🎯 เริ่มระบบเว็บ Enhanced..."
        echo "URL: http://localhost:8501"
        streamlit run ai_agents_web_enhanced.py --server.port 8501
        ;;
    3)
        echo "🔧 เริ่มระบบเว็บ Clean..."
        echo "URL: http://localhost:8502"
        streamlit run ai_agents_web_enhanced_clean.py --server.port 8502
        ;;
    4)
        echo "📱 เริ่มระบบเว็บ Basic..."
        echo "URL: http://localhost:8504"
        streamlit run ai_agents_web.py --server.port 8504
        ;;
    5)
        echo "💻 เริ่ม CLI..."
        python run_ai_agents_simple.py --action analyze --verbose
        ;;
    *)
        echo "❌ ตัวเลือกไม่ถูกต้อง"
        echo "🚀 เริ่มระบบเว็บ Ultimate (default)..."
        streamlit run ai_agents_web_ultimate.py --server.port 8503
        ;;
esac
