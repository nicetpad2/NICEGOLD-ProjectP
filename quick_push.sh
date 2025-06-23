#!/bin/bash
# -*- coding: utf-8 -*-
# 🚀 NICEGOLD Quick Push Script
# สคริปต์สำหรับ push การเปลี่ยนแปลงไปยัง GitHub repository

set -e  # หยุดเมื่อมี error

# สี ANSI สำหรับ output ที่สวยงาม
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ฟังก์ชันสำหรับแสดงข้อความ
print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}🚀 NICEGOLD GitHub Push Manager${NC}"
    echo -e "${PURPLE}================================${NC}"
}

print_step() {
    echo -e "${CYAN}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ตรวจสอบว่าอยู่ในไดเรกทอรีที่ถูกต้อง
check_directory() {
    if [[ ! -d ".git" ]]; then
        print_error "ไม่พบไดเรกทอรี .git - โปรดรันสคริปต์ในโฟลเดอร์ repository"
        exit 1
    fi
    
    if [[ ! -f "ADMIN_GUIDE.md" ]]; then
        print_warning "ไม่พบไฟล์ ADMIN_GUIDE.md - อาจไม่ใช่โฟลเดอร์ NICEGOLD"
    fi
}

# ตั้งค่า Git user (ถ้ายังไม่ได้ตั้ง)
setup_git_user() {
    print_step "ตรวจสอบการตั้งค่า Git user..."
    
    if ! git config user.name > /dev/null 2>&1; then
        git config user.name "NICEGOLD Administrator"
        print_success "ตั้งค่า Git username: NICEGOLD Administrator"
    fi
    
    if ! git config user.email > /dev/null 2>&1; then
        git config user.email "admin@nicegold.local"
        print_success "ตั้งค่า Git email: admin@nicegold.local"
    fi
}

# ตรวจสอบสถานะ Git
check_git_status() {
    print_step "ตรวจสอบสถานะ repository..."
    
    # ตรวจสอบ branch ปัจจุบัน
    CURRENT_BRANCH=$(git branch --show-current)
    print_success "Branch ปัจจุบัน: $CURRENT_BRANCH"
    
    # ตรวจสอบ remote
    if git remote get-url origin > /dev/null 2>&1; then
        REMOTE_URL=$(git remote get-url origin)
        print_success "Remote URL: $REMOTE_URL"
    else
        print_error "ไม่พบ remote origin"
        exit 1
    fi
    
    # ตรวจสอบไฟล์ที่เปลี่ยนแปลง
    if [[ -n $(git status --porcelain) ]]; then
        print_success "พบไฟล์ที่เปลี่ยนแปลง:"
        git status --short
    else
        print_warning "ไม่พบไฟล์ที่เปลี่ยนแปลง"
        echo -e "${YELLOW}คุณต้องการ push แบบ force หรือไม่? (y/n)${NC}"
        read -r FORCE_PUSH
        if [[ $FORCE_PUSH != "y" && $FORCE_PUSH != "Y" ]]; then
            print_success "ยกเลิกการ push"
            exit 0
        fi
    fi
}

# เพิ่มไฟล์เข้า staging area
add_files() {
    print_step "เพิ่มไฟล์เข้า staging area..."
    
    # ลบไฟล์ขยะก่อน
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".DS_Store" -delete 2>/dev/null || true
    
    # เพิ่มไฟล์ทั้งหมด
    git add .
    
    print_success "เพิ่มไฟล์ทั้งหมดเรียบร้อย"
    
    # แสดงไฟล์ที่จะ commit
    if [[ -n $(git diff --cached --name-only) ]]; then
        print_success "ไฟล์ที่จะ commit:"
        git diff --cached --name-only | head -20
        
        TOTAL_FILES=$(git diff --cached --name-only | wc -l)
        if [[ $TOTAL_FILES -gt 20 ]]; then
            print_warning "... และอีก $(($TOTAL_FILES - 20)) ไฟล์"
        fi
    fi
}

# สร้าง commit message
create_commit_message() {
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [[ -z $1 ]]; then
        COMMIT_MESSAGE="🚀 NICEGOLD Enterprise Update - $TIMESTAMP"
    else
        COMMIT_MESSAGE="$1"
    fi
    
    # นับไฟล์ที่เปลี่ยนแปลง
    CHANGED_FILES=$(git diff --cached --name-only | wc -l)
    
    COMMIT_DESCRIPTION="📝 Updated $CHANGED_FILES files in NICEGOLD Enterprise system

🔧 Changes include:
- System improvements and bug fixes
- Configuration updates
- Documentation updates
- Feature enhancements

⚡ Auto-generated commit from NICEGOLD push script
🕒 Timestamp: $TIMESTAMP"

    echo "$COMMIT_MESSAGE" > /tmp/commit_msg.txt
    echo "" >> /tmp/commit_msg.txt
    echo "$COMMIT_DESCRIPTION" >> /tmp/commit_msg.txt
}

# Commit การเปลี่ยนแปลง
commit_changes() {
    print_step "กำลัง commit การเปลี่ยนแปลง..."
    
    create_commit_message "$1"
    
    if git commit -F /tmp/commit_msg.txt; then
        print_success "Commit สำเร็จ!"
        rm -f /tmp/commit_msg.txt
    else
        print_error "Commit ล้มเหลว"
        rm -f /tmp/commit_msg.txt
        exit 1
    fi
}

# Sync กับ remote
sync_with_remote() {
    print_step "ซิงค์กับ remote repository..."
    
    # Fetch ข้อมูลล่าสุด
    if git fetch origin; then
        print_success "Fetch สำเร็จ"
    else
        print_warning "Fetch ล้มเหลว - ข้ามขั้นตอนนี้"
    fi
    
    # ตรวจสอบว่ามี commits ใหม่จาก remote หรือไม่
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "")
    
    if [[ -n $REMOTE && $LOCAL != $REMOTE ]]; then
        print_step "พบการเปลี่ยนแปลงจาก remote - กำลัง pull..."
        if git pull origin $(git branch --show-current); then
            print_success "Pull สำเร็จ"
        else
            print_warning "Pull ล้มเหลว - อาจมี conflict"
            print_warning "โปรดแก้ไข conflict ด้วยตนเอง"
            exit 1
        fi
    fi
}

# Push ไปยัง remote
push_to_remote() {
    print_step "กำลัง push ไปยัง remote repository..."
    
    CURRENT_BRANCH=$(git branch --show-current)
    
    if git push origin $CURRENT_BRANCH; then
        print_success "Push สำเร็จ! 🎉"
        
        # แสดง URL ของ repository
        REMOTE_URL=$(git remote get-url origin)
        if [[ $REMOTE_URL == *"github.com"* ]]; then
            REPO_URL=$(echo $REMOTE_URL | sed 's/\.git$//' | sed 's/git@github\.com:/https:\/\/github.com\//')
            print_success "ดู repository ได้ที่: $REPO_URL"
        fi
    else
        print_error "Push ล้มเหลว"
        
        print_warning "ลองใช้ force push หรือไม่? (y/n)"
        read -r USE_FORCE
        
        if [[ $USE_FORCE == "y" || $USE_FORCE == "Y" ]]; then
            print_step "กำลัง force push..."
            if git push --force-with-lease origin $CURRENT_BRANCH; then
                print_success "Force push สำเร็จ! 🎉"
            else
                print_error "Force push ล้มเหลว"
                exit 1
            fi
        else
            exit 1
        fi
    fi
}

# ฟังก์ชันหลัก
main() {
    print_header
    
    check_directory
    setup_git_user
    check_git_status
    add_files
    commit_changes "$1"
    sync_with_remote
    push_to_remote
    
    print_success "การ push เสร็จสิ้น! 🚀"
}

# เรียกใช้ฟังก์ชันหลัก
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$1"
fi
