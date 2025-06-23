#!/bin/bash
# Git Commit Script สำหรับโปรเจค ML Pipeline
# สร้างโดย: Git File Analysis Tool
# วันที่: $(date)

echo "🚀 เริ่มต้นการ commit ไฟล์โปรเจค"
echo "================================="

# ขั้นที่ 1: Commit ไฟล์สำคัญมาก
echo "📋 ขั้นที่ 1: Commit ไฟล์สำคัญมาก"

# ไฟล์สำคัญมาก
git add "README.md"
git add "README_REFACTORED.md"
git add "README_SETUP.md"
git add "README_TRACKING.md"
git add "README_ULTIMATE_FIX.md"
git add "ProjectP.py"
git add "advanced_protection_auto_setup.py"
git add "colab_auto_setup.py"
git add "environment_setup.py"
git add "fast_projectp.py"
git add "main.py"
git add "safe_projectp.py"
git add "setup.py"
git commit -m "feat: เพิ่มไฟล์หลักของโปรเจค (Core Files)"
echo "✅ Commit ไฟล์สำคัญมากเสร็จสิ้น"

# ขั้นที่ 2: Commit ไฟล์ Python สำคัญ
echo "📋 ขั้นที่ 2: Commit ไฟล์ Python สำคัญ"
git add "Dockerfile"
git add "PYDANTIC_V2_FIX_GUIDE.py"
git add "docker-compose.yml"
git add "advanced_ml_protection_config.yaml"
git add "agent_config.yaml"
git add "config.yaml"
git add "config_loader.py"
git add "logging_config.yaml"
git add "ml_protection_config.yaml"
git add "ml_protection_config_backup.yaml"
git add "monitoring_config.yaml"
git add "tracking_config.yaml"
git add "AUC_SOLUTION_SUMMARY.py"
git add "ProjectP_Complete.py"
git add "ProjectP_refactored.py"
# ... และอีก 190 ไฟล์
git commit -m "feat: เพิ่มไฟล์ Python หลักของโปรเจค"
echo "✅ Commit ไฟล์ Python สำคัญเสร็จสิ้น"

# ขั้นที่ 3: Commit ไฟล์ Configuration
echo "📋 ขั้นที่ 3: Commit ไฟล์ Configuration"
git add "advanced_ml_protection_config.yaml"
git add "agent_config.yaml"
git add "config.yaml"
git add "config_loader.py"
git add "logging_config.yaml"
git add "ml_protection_config.yaml"
git add "ml_protection_config_backup.yaml"
git add "monitoring_config.yaml"
git add "tracking_config.yaml"
git commit -m "config: เพิ่มไฟล์ configuration และ setup"
echo "✅ Commit ไฟล์ Configuration เสร็จสิ้น"

# ขั้นที่ 4: Commit เอกสาร
echo "📋 ขั้นที่ 4: Commit เอกสาร"
git add "ADVANCED_ML_PROTECTION_COMPLETE_GUIDE.md"
git add "AGENTS.md"
git add "CHANGELOG.md"
git add "COMPLETE_FIX_SUMMARY.md"
git add "COMPLETE_RESOLUTION_REPORT.md"
git add "DEPLOYMENT_GUIDE.md"
git add "ENHANCED_AGENT_SUCCESS_REPORT.md"
git add "FINAL_PRODUCTION_SUCCESS_REPORT.md"
git add "FINAL_PRODUCTION_SUCCESS_REPORT_TH.md"
git add "FINAL_SOLUTION_SUMMARY.md"
# ... และอีก 33 ไฟล์
git commit -m "docs: เพิ่มเอกสารและคู่มือโปรเจค"
echo "✅ Commit เอกสารเสร็จสิ้น"


# สรุปผลการ commit
echo "🎉 การ commit เสร็จสิ้น!"
echo "========================"
echo "📊 สถิติ Git Repository:"
git log --oneline | head -5
echo ""
echo "📁 ไฟล์ที่ถูก track:"
git ls-files | wc -l
echo " ไฟล์"
echo ""
echo "✅ พร้อมสำหรับการ push ขึ้น remote repository"
