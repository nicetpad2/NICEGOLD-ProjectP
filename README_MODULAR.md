# NICEGOLD ProjectP v2.0 - Modular Architecture

## 🚀 Professional AI Trading System

ระบบ AI Trading แบบ Professional ที่ได้รับการพัฒนาใหม่ด้วย Modular Architecture เพื่อความง่ายในการบำรุงรักษาและการพัฒนา

### 📁 โครงสร้างโปรเจค

```
NICEGOLD-ProjectP/
├── ProjectP.py                  # Main entry point (ไฟล์หลัก)
├── core/                        # Core modules (โมดูลหลัก)
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── system.py               # System utilities
│   ├── menu_operations.py      # Menu implementations
│   └── menu_interface.py       # User interface
├── utils/                       # Utility modules (โมดูลช่วยเหลือ)
│   ├── __init__.py
│   └── colors.py               # Color formatting
├── src/                        # Source code (โค้ดหลัก)
│   ├── core/                   # Core functionality
│   ├── ml/                     # Machine learning
│   ├── trading/                # Trading logic
│   └── utils/                  # Utilities
├── datacsv/                    # Input data folder (โฟลเดอร์ข้อมูล)
├── output_default/             # Results folder (โฟลเดอร์ผลลัพธ์)
├── models/                     # Trained models (โมเดลที่ฝึกแล้ว)
├── logs/                       # Log files (ไฟล์ log)
├── config.yaml                 # Configuration file (ไฟล์ config)
└── README.md                   # This file
```

### 🛠️ การติดตั้ง

#### 1. **Quick Setup (การติดตั้งแบบเร็ว):**
```bash
python3 ProjectP.py
# เลือกตัวเลือก 8 เพื่อติดตั้ง dependencies
```

#### 2. **Manual Installation (การติดตั้งแบบ manual):**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib pyyaml tqdm requests streamlit catboost lightgbm xgboost optuna shap ta
```

### 🚀 การใช้งาน

#### 1. **เริ่มต้นแอปพลิเคชัน:**
```bash
python3 ProjectP.py
```

#### 2. **เพิ่มข้อมูลของคุณ:**
- วางไฟล์ CSV ลงในโฟลเดอร์ `datacsv/`
- ตรวจสอบให้แน่ใจว่าข้อมูลมี columns: Open, High, Low, Close, Volume

#### 3. **ใช้งานผ่านเมนู:**
เลือกตัวเลือกที่ต้องการจากเมนูหลัก

### 🔧 ฟีเจอร์หลัก

#### 📊 **เมนูตัวเลือก**

| ตัวเลือก | ฟีเจอร์ | คำอธิบาย |
|--------|---------|-------------|
| 1 | 🚀 Full Pipeline | รัน ML trading workflow แบบสมบูรณ์ |
| 2 | 📊 Data Analysis | วิเคราะห์ข้อมูลแบบครอบคลุม |
| 3 | 🔧 Quick Test | ทดสอบการทำงานของระบบ |
| 4 | 🤖 Train Models | ฝึกโมเดล Machine Learning |
| 5 | 📈 Backtest | ทดสอบกลยุทธ์การเทรด |
| 6 | 🌐 Web Dashboard | เปิด Streamlit web interface |
| 7 | 🔍 Health Check | ตรวจสอบสถานะระบบ |
| 8 | 📦 Install Dependencies | ติดตั้ง packages ที่จำเป็น |
| 9 | 🧹 Clean System | ทำความสะอาดไฟล์ temporary |
| 0 | 👋 Exit | ออกจากโปรแกรม |

#### 🧠 **Advanced Features**
- **Modular Design:** แยกเป็นโมดูลตามหน้าที่ เพื่อความง่ายในการพัฒนา
- **Easy Configuration:** ใช้ YAML สำหรับการตั้งค่า
- **Multiple ML Models:** รองรับ Random Forest, XGBoost, LightGBM, CatBoost
- **Web Interface:** Dashboard แบบ Streamlit
- **System Monitoring:** ตรวจสอบสุขภาพระบบ built-in
- **Error Handling:** จัดการ error แบบ robust
- **Beautiful UI:** Interface แบบ colorized terminal
- **Fallback Mode:** ทำงานได้แม้ว่า dependencies บางตัวจะไม่มี

### 🔄 การพัฒนา

Architecture ใหม่ทำให้ง่ายต่อการ:

#### **เพิ่มฟีเจอร์ใหม่:**
- แก้ไขไฟล์ `core/menu_operations.py` เพื่อเพิ่ม menu options
- เพิ่ม functions ใน `core/` modules

#### **ปรับแต่ง UI:**
- แก้ไขไฟล์ `core/menu_interface.py`
- ปรับแต่ง colors ใน `utils/colors.py`

#### **เปลี่ยนการตั้งค่า:**
- แก้ไขไฟล์ `config.yaml`
- ใช้ `core.config` module เพื่อจัดการ config

#### **เพิ่ม utilities:**
- เพิ่มโมดูลใน `utils/` หรือ `src/`

### 🐛 การแก้ไขปัญหา

#### 1. **Import errors:**
```bash
python3 ProjectP.py
# เลือกตัวเลือก 8 เพื่อติดตั้ง dependencies
```

#### 2. **Missing packages:**
- ตรวจสอบ requirements และติดตั้งแบบ manual
- ใช้ `pip install package_name`

#### 3. **Permission errors:**
- ตรวจสอบสิทธิ์ในการเขียนไฟล์ในโฟลเดอร์โปรเจค

#### 4. **Data issues:**
- ตรวจสอบรูปแบบ CSV ในโฟลเดอร์ `datacsv/`
- ต้องมี columns: Open, High, Low, Close, Volume

### 📝 การตั้งค่า

#### แก้ไขไฟล์ `config.yaml`:

```yaml
project:
  name: "NICEGOLD ProjectP"
  version: "2.0"
  description: "Professional AI Trading System"

data:
  input_folder: "datacsv"
  output_folder: "output_default"
  models_folder: "models"
  logs_folder: "logs"

trading:
  default_balance: 10000
  risk_per_trade: 0.02
  max_positions: 5
  commission: 0.001

ml:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
  models: ["RandomForest", "LightGBM", "CatBoost"]

ui:
  show_animations: true
  color_output: true
  menu_timeout: 300
```

### ✅ ข้อดีของ Architecture ใหม่

#### **สำหรับผู้ใช้:**
- **ใช้งานง่าย:** Interface ที่ใช้งานง่ายและสวยงาม
- **เสถียร:** Error handling ที่ดีกว่า
- **ยืดหยุ่น:** ทำงานได้แม้ว่า dependencies บางตัวจะไม่มี
- **ครบถ้วน:** ฟีเจอร์ครบครันสำหรับ AI Trading

#### **สำหรับนักพัฒนา:**
- **Maintainable:** แยกส่วนงานอย่างชัดเจน (Separation of Concerns)
- **Extensible:** เพิ่มฟีเจอร์ใหม่ได้ง่าย
- **Testable:** โมดูลแยกจากกัน ทดสอบได้ง่าย
- **Readable:** โค้ดเข้าใจง่าย มี documentation ครบ
- **Scalable:** ขยายตามความต้องการได้

### 📋 System Requirements

- **Python:** 3.8 หรือสูงกว่า
- **OS:** Windows, macOS, Linux
- **RAM:** 4GB ขั้นต่ำ (แนะนำ 8GB)
- **Storage:** 2GB ว่างสำหรับข้อมูลและโมเดล

### 📞 การสนับสนุน

หากพบปัญหาหรือต้องการความช่วยเหลือ:

1. ตรวจสอบ logs ในโฟลเดอร์ `logs/`
2. ใช้ตัวเลือก 7 (Health Check) เพื่อตรวจสอบสถานะระบบ
3. ใช้ตัวเลือก 9 (Clean System) เพื่อทำความสะอาดระบบ

### 🎯 เป้าหมายในอนาคต

- [ ] เพิ่ม Real-time data feeds
- [ ] รองรับ Multi-asset trading
- [ ] เพิ่ม Advanced risk management
- [ ] พัฒนา Mobile app
- [ ] เพิ่ม Social trading features

---

**NICEGOLD ProjectP v2.0** - Professional AI Trading System with Modular Architecture

*สร้างโดย NICEGOLD Enterprise*  
*วันที่: 24 มิถุนายน 2025*
