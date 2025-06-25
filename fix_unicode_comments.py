#!/usr/bin/env python3
"""
Script to fix all Unicode/Thai comments in config.py for production compatibility
"""

import os
import re


def fix_unicode_comments():
    """Replace all Thai/Unicode comments with ASCII equivalents"""

    config_file = "src/config.py"

    if not os.path.exists(config_file):
        print(f"File {config_file} not found!")
        return

    # Read the file
    with open(config_file, "r", encoding = "utf - 8") as f:
        content = f.read()

    # Define replacement patterns
    replacements = [
        # Thai comments to ASCII
        (r"# กำหนดค่าพื้นฐานสำหรับการ Logging", "# Basic logging configuration"), 
        (
            r"# จัดเก็บไฟล์ log ลงในโฟลเดอร์ย่อยตามวันที่และ fold", 
            "# Store log files in subdirectories by date and fold", 
        ), 
        (
            r"# ตั้งค่า Logger กลางเพื่อให้โมดูลอื่น ๆ ใช้งานร่วมกัน", 
            "# Setup central logger for shared module usage", 
        ), 
        (
            r"# รองรับโหมด COMPACT_LOG เพื่อลดข้อความที่แสดงบนหน้าจอ", 
            "# Support COMPACT_LOG mode to reduce screen output", 
        ), 
        (
            r"# ยืนยันว่าฟังก์ชันใช้ตัวแปร logger ที่นำเข้าไว้ด้านบน", 
            "# Confirm function uses logger variable imported above", 
        ), 
        (
            r'logging\.info\("   กำลังติดตั้งไลบรารี ([^"] + )\.\.\."\)', 
            r'logging.info("   Installing \1 library...")', 
        ), 
        (
            r'logging\.info\("   \(Success\) ติดตั้ง ([^"] + ) สำเร็จ\."\)', 
            r'logging.info("   (Success) \1 installed successfully.")', 
        ), 
        (
            r'logging\.error\(f"   \(Error\) ไม่สามารถติดตั้ง ([^"] + ): \{([^}] + )\}', 
            r'logging.error(f"   (Error) Cannot install \1: {\2}', 
        ), 
        (
            r'logging\.warning\("ไลบรารี \'([^\'] + )\' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS = False', 
            r'logging.warning("Library \'\1\' not installed and AUTO_INSTALL_LIBS = False', 
        ), 
        (
            r'"   \(Info\) ตรวจสอบเวอร์ชัน ([^"] + ): "', 
            r'"   (Info) Checking \1 version: "', 
        ), 
        (
            r'"   \(Info\) ตรวจสอบจำนวน GPU สำหรับ ([^"] + ): "', 
            r'"   (Info) Checking GPU count for \1: "', 
        ), 
        (
            r'"   \(Warning\) ไม่สามารถตรวจสอบจำนวน GPU ของ ([^"] + ): "', 
            r'"   (Warning) Cannot check GPU count for \1: "', 
        ), 
        (r'"   ผลการติดตั้ง ([^"] + ): \.\.\."', r'"   \1 installation result: ..."'), 
        (
            r'"\(Info\) รันบน Google Colab – กำลัง mount Google Drive\.\.\."', 
            '"(Info) Running on Google Colab - mounting Google Drive..."', 
        ), 
        (
            r'"\(Success\) Mount Google Drive สำเร็จ"', 
            '"(Success) Google Drive mounted successfully"', 
        ), 
        (
            r'"\(Info\) ไม่ใช่ Colab – ใช้เส้นทางในเครื่องสำหรับจัดเก็บ log และข้อมูล"', 
            '"(Info) Not Colab - using local paths for logs and data storage"', 
        ), 
        (
            r"# ใช้โฟลเดอร์ Google Drive หากมีให้บริการแม้ไม่อยู่ใน Colab", 
            "# Use Google Drive folder if available even outside Colab", 
        ), 
        (
            r"# ใช้โฟลเดอร์ปัจจุบันเป็นฐานข้อมูลเพื่อให้ทำงานได้ทุกที่", 
            "# Use current folder as base to work everywhere", 
        ), 
        (
            r'"   การติดตั้ง SHAP อาจใช้เวลาสักครู่\.\.\."', 
            '"   SHAP installation may take a while..."', 
        ), 
        (
            r'"   กำลังติดตั้ง GPUtil สำหรับตรวจสอบ GPU \(Optional\)\.\.\."', 
            '"   Installing GPUtil for GPU monitoring (Optional)..."', 
        ), 
        (
            r'"   \(Warning\) ไม่สามารถติดตั้ง GPUtil: ([^"] + )\. ฟังก์ชัน show_system_status อาจไม่ทำงาน\."', 
            r'"   (Warning) Cannot install GPUtil: \1. show_system_status function may not work."', 
        ), 
        (
            r'"ไลบรารี \'GPUtil\' ไม่ถูกติดตั้ง หรือไม่สามารถโหลดได้"', 
            "\"Library 'GPUtil' not installed or cannot be loaded\"", 
        ), 
        # Specific logging messages
        (
            r'logger\.info\(" -  - - กำลังโหลดไลบรารีและตรวจสอบ Dependencies - -  - "\)', 
            'logger.info(" -  - - Loading libraries and checking dependencies - -  - ")', 
        ), 
        (
            r"# minimum rows required for trade log validation", 
            "# minimum rows required for trade log validation", 
        ), 
        # Remove any remaining emoji/special Unicode characters in logging
        (r"🔍", "[CHECK]"), 
        (r"✅", "[OK]"), 
        (r"⚠️", "[WARNING]"), 
        (r"💻", "[SYSTEM]"), 
        (r"🎯", "[TARGET]"), 
        (r"📦", "[PACKAGE]"), 
        (r"📊", "[DATA]"), 
        (r"🚀", "[LAUNCH]"), 
        (r"ℹ️", "[INFO]"), 
        (r"❌", "[ERROR]"), 
        (r"💡", "[IDEA]"), 
        (r"🔧", "[TOOL]"), 
        (r"📈", "[CHART]"), 
        (r"📉", "[TREND]"), 
        (r"⭐", "[STAR]"), 
        (r"🎉", "[SUCCESS]"), 
        (r"🔥", "[HOT]"), 
        (r"⚡", "[FAST]"), 
        (r"🌟", "[HIGHLIGHT]"), 
        (r"💎", "[PREMIUM]"), 
        (r"🏆", "[WINNER]"), 
        (r"🎨", "[DESIGN]"), 
        (r"🔒", "[SECURE]"), 
    ]

    # Apply all replacements
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags = re.MULTILINE)

    # Additional cleanup - remove any remaining non - ASCII characters in log messages
    # This is a more aggressive approach for any remaining Unicode in strings
    lines = content.split("\n")
    cleaned_lines = []

    for line in lines:
        # If line contains logging and has non - ASCII characters, try to clean it
        if any(keyword in line for keyword in ["logger.", "logging."]) and any(
            ord(char) > 127 for char in line
        ):
            # Replace any remaining Thai characters with [THAI] placeholder
            line = re.sub(r"[ก - ๙] + ", "[THAI]", line)
            # Replace any other high Unicode with [UNICODE]
            line = re.sub(r"[^\x00 - \x7F] + ", "[UNICODE]", line)
        cleaned_lines.append(line)

    content = "\n".join(cleaned_lines)

    # Write back to file
    with open(config_file, "w", encoding = "utf - 8") as f:
        f.write(content)

    print(f"✓ Fixed Unicode comments in {config_file}")
    print(
        "All Unicode/emoji characters in logging messages have been replaced with ASCII equivalents"
    )


if __name__ == "__main__":
    fix_unicode_comments()