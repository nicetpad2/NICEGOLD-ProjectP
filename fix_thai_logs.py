#!/usr/bin/env python3
"""
Script to replace all Thai logging messages with English equivalents
for production-ready cross-platform compatibility.
"""

import re


def fix_thai_logs():
    config_file = "src/config.py"

    # Read the file
    with open(config_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Define Thai to English replacements for logging messages
    replacements = [
        # Thai logging messages
        (r"ไม่สามารถติดตั้ง", "Cannot install"),
        (r"จะไม่ทำงาน", "will not work"),
        (r"ไลบรารี", "Library"),
        (r"ไม่ถูกติดตั้ง", "not installed"),
        (r"ข้ามการ", "skipping"),
        (r"ข้ามขั้นตอน", "skipping step"),
        (r"การติดตั้ง", "Installation"),
        (r"อาจใช้เวลาสักครู่", "may take a moment"),
        (r"ติดตั้ง.*สำเร็จ", "installed successfully"),
        (r"เวอร์ชัน", "version"),
        (r"ตรวจสอบ", "checking"),
        (r"จำนวน", "number of"),
        (r"สำหรับ", "for"),
        (r"หลังติดตั้ง", "after installation"),
        (r"การวิเคราะห์", "analysis"),
        (r"จะถูกข้ามไป", "will be skipped"),
        (r"การคำนวณ", "calculation"),
        (r"ไม่สามารถ", "cannot"),
        (r"หรือไม่สามารถโหลดได้", "or cannot be loaded"),
        (r"ฟังก์ชัน.*อาจไม่ทำงาน", "function may not work"),
        # Specific multi-word phrases
        (
            r"ไม่สามารถติดตั้ง optuna.*Hyperparameter Optimization จะไม่ทำงาน",
            "Cannot install optuna. Hyperparameter Optimization will not work",
        ),
        (
            r"ไลบรารี \'optuna\' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS=False.*ข้ามการปรับแต่ง",
            "Library 'optuna' not installed and AUTO_INSTALL_LIBS=False -- skipping optimization",
        ),
        (
            r"ไม่สามารถติดตั้ง catboost.*CatBoost models และ SHAP อาจไม่ทำงาน",
            "Cannot install catboost. CatBoost models and SHAP may not work",
        ),
        (
            r"ไลบรารี \'catboost\' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS=False.*ข้ามขั้นตอน CatBoost",
            "Library 'catboost' not installed and AUTO_INSTALL_LIBS=False -- skipping CatBoost step",
        ),
        (
            r"ไม่สามารถติดตั้ง shap.*การวิเคราะห์ SHAP จะถูกข้ามไป",
            "Cannot install shap. SHAP analysis will be skipped",
        ),
        (
            r"ไลบรารี \'shap\' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS=False.*ข้ามการคำนวณ SHAP",
            "Library 'shap' not installed and AUTO_INSTALL_LIBS=False -- skipping SHAP calculation",
        ),
        # Comments and simple strings
        (r"กำลังติดตั้งไลบรารี", "Installing library"),
        (r"กำลังติดตั้ง", "Installing"),
        (r"ผลการติดตั้ง", "Installation result"),
        (r"ไม่พบ กำลังติดตั้งอัตโนมัติ", "not found, installing automatically"),
        (r"ไม่สามารถตรวจสอบจำนวน GPU ของ", "Cannot check GPU count for"),
        (r"ตรวจสอบเวอร์ชัน", "Checking version"),
        (r"ตรวจสอบจำนวน GPU สำหรับ", "Checking GPU count for"),
        # File paths and technical terms
        (
            r"--- กำลังโหลดไลบรารีและตรวจสอบ Dependencies ---",
            "--- Loading libraries and checking dependencies ---",
        ),
        (r"--- \(Start\) Gold AI v.*---", "--- (Start) Gold AI v{version} ---"),
    ]

    # Apply all replacements
    for thai_pattern, english_replacement in replacements:
        content = re.sub(
            thai_pattern, english_replacement, content, flags=re.IGNORECASE
        )

    # Write back the file
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(content)

    print("Fixed Thai logging messages in config.py")


if __name__ == "__main__":
    fix_thai_logs()
