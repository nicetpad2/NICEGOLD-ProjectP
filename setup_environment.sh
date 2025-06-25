#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 🚀 NICEGOLD ProjectP Environment Setup Script
# ═══════════════════════════════════════════════════════════════

echo "🔧 กำลังตั้งค่า environment สำหรับ NICEGOLD ProjectP..."

# โหลด environment variables สำหรับ pip และ Python
export PIP_CACHE_DIR="/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/.cache/pip"
export TMPDIR="/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/.tmp"
export TEMP="/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/.tmp"
export TMP="/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/.tmp"

# เปิดใช้งาน virtual environment
source /home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/.venv/bin/activate

echo "✅ Environment ตั้งค่าเรียบร้อย!"
echo "📂 PIP_CACHE_DIR: $PIP_CACHE_DIR"
echo "📂 TMPDIR: $TMPDIR"
echo "🐍 Python Virtual Environment: Activated"
echo ""
echo "🎯 พร้อมใช้งาน NICEGOLD ProjectP"
echo "💡 คำแนะนำ: ใช้คำสั่ง 'source setup_environment.sh' ก่อนเริ่มงาน"
