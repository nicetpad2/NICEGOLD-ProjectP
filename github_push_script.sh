#!/bin/bash
# GitHub Push Script สำหรับโปรเจค ML Pipeline
# สร้างโดย: Git File Analysis Tool
# วันที่: $(date)

echo "🚀 เตรียมการ Push ขึ้น GitHub"
echo "================================"

# ตรวจสอบสถานะปัจจุบัน
echo "📊 ตรวจสอบสถานะ Git Repository:"
git status --short

echo ""
echo "📝 Commit History:"
git log --oneline -n 5

echo ""
echo "📁 ไฟล์ที่ถูก track:"
git ls-files | wc -l
echo " ไฟล์"

echo ""
echo "🔧 คำสั่งสำหรับ Push ขึ้น GitHub:"
echo "================================"
echo ""
echo "1. สร้าง Repository บน GitHub แล้วคัดลอกคำสั่งด้านล่าง:"
echo ""
echo "# เชื่อมต่อกับ GitHub Repository"
echo "git remote add origin https://github.com/[username]/[repository-name].git"
echo ""
echo "# เปลี่ยน branch เป็น main"
echo "git branch -M main"
echo ""
echo "# Push ขึ้น GitHub"
echo "git push -u origin main"
echo ""
echo "2. แทนที่ [username] และ [repository-name] ด้วยข้อมูลจริง"
echo ""
echo "3. ตัวอย่าง:"
echo "   git remote add origin https://github.com/myusername/ml-pipeline-project.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "✅ Repository พร้อมสำหรับการแชร์บน GitHub!"
