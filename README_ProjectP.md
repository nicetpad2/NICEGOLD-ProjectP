# 🚀 NICEGOLD ProjectP - Main Launcher

## การใช้งาน ProjectP.py

ไฟล์ `ProjectP.py` เป็นไฟล์หลักสำหรับการรันระบบ NICEGOLD ProjectP แบบ Interactive Menu

### วิธีการเริ่มต้น

#### วิธีที่ 1: ใช้ Quick Launcher
```bash
./start.sh
```

#### วิธีที่ 2: รันโดยตรง
```bash
source setup_environment.sh
python ProjectP.py
```

#### วิธีที่ 3: รันแบบ Manual
```bash
# ตั้งค่า environment
export PIP_CACHE_DIR="/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/.cache/pip"
export TMPDIR="/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/.tmp"

# เปิดใช้งาน virtual environment
source .venv/bin/activate

# รันโปรแกรม
python ProjectP.py
```

## เมนูหลักของ ProjectP

### 🚀 Core Pipeline Modes:
- **1. Full Pipeline** - รันระบบครบทุกขั้นตอน (Production Ready)
- **2. Debug Pipeline** - โหมดดีบัก: ตรวจสอบทุกจุด (Detailed Logs)
- **3. Quick Test** - ทดสอบเร็ว: ข้อมูลย่อย (Development)

### 📊 Data Processing:
- **4. Load & Validate Data** - โหลดและตรวจสอบข้อมูลจริง
- **5. Feature Engineering** - สร้าง Technical Indicators
- **6. Preprocess Only** - เตรียมข้อมูลสำหรับ ML

### 🤖 Machine Learning:
- **7. Train Models** - เทรนโมเดล ML (AutoML + Optimization)
- **8. Model Comparison** - เปรียบเทียบโมเดลต่างๆ
- **9. Predict & Backtest** - ทำนายและ Backtest

### 📈 Advanced Analytics:
- **10. Live Trading Simulation** - จำลองการเทรดแบบ Real-time
- **11. Performance Analysis** - วิเคราะห์ผลงานแบบละเอียด
- **12. Risk Management** - จัดการความเสี่ยงและ Portfolio

### 🖥️ Monitoring & Services:
- **13. Web Dashboard** - เปิด Streamlit Dashboard
- **14. API Server** - เปิด FastAPI Model Server
- **15. Real-time Monitor** - ติดตามระบบแบบ Real-time

### ⚙️ System Management:
- **16. System Health Check** - ตรวจสอบสุขภาพระบบทั้งหมด
- **17. Install Dependencies** - ติดตั้งไลบรารี่ที่จำเป็น
- **18. Clean & Reset** - ล้างข้อมูลและรีเซ็ตระบบ
- **19. View Logs & Results** - ดูผลลัพธ์และ Log Files

## คุณสมบัติพิเศษ

### 🔍 System Health Check
- ตรวจสอบไลบรารี่ Python ทั้งหมด
- ตรวจสอบไฟล์ข้อมูล (XAUUSD_M1.csv, XAUUSD_M15.csv)
- ตรวจสอบโฟลเดอร์ที่จำเป็น
- แสดงสถานะระบบแบบละเอียด

### 🎯 Smart Command Execution
- รันคำสั่งด้วย Python interpreter ที่ถูกต้อง
- จัดการ environment variables อัตโนมัติ
- Error handling และ logging ที่ครบถ้วน
- รองรับการรันแบบ background และ foreground

### 📝 Comprehensive Logging
- บันทึก log ลงในไฟล์
- แสดงผลลัพธ์แบบ Real-time
- รองรับการดู log ย้อนหลัง

## การแก้ปัญหา

### ปัญหาที่พบบ่อย

#### 1. ไลบรารี่ขาดหายไป
```bash
# เลือกเมนู 17 หรือรันคำสั่ง
pip install -r requirements.txt
```

#### 2. Environment variables ไม่ถูกต้อง
```bash
source setup_environment.sh
```

#### 3. ข้อมูลไม่พบ
```bash
# ตรวจสอบไฟล์ข้อมูลใน datacsv/
ls -la datacsv/
```

#### 4. พื้นที่ดิสก์เต็ม
```bash
# ใช้เมนู 18 เพื่อล้างข้อมูล
# หรือตรวจสอบพื้นที่
df -h
```

## ไฟล์ที่เกี่ยวข้อง

- `ProjectP.py` - ไฟล์หลัก (Interactive Menu)
- `main.py` - Pipeline หลัก
- `config.yaml` - การตั้งค่าระบบ
- `setup_environment.sh` - ตั้งค่า environment
- `start.sh` - Quick launcher
- `requirements.txt` - รายการไลบรารี่

## การพัฒนาเพิ่มเติม

### เพิ่มเมนูใหม่
1. แก้ไขฟังก์ชัน `print_main_menu()` 
2. เพิ่ม case ใหม่ในฟังก์ชัน `handle_menu_choice()`
3. เพิ่มการเรียกใช้ในส่วน implementation

### เพิ่มการตรวจสอบระบบ
1. แก้ไขฟังก์ชัน `check_system_health()`
2. เพิ่มการตรวจสอบไฟล์/โฟลเดอร์/package ใหม่
3. อัปเดตการแสดงผลใน `display_system_status()`

---

🎯 **NICEGOLD ProjectP** - Professional AI Trading System  
💡 สำหรับคำถามหรือปัญหา กรุณาตรวจสอบ log files ใน `output_default/logs/`
