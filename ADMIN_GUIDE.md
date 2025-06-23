# 🚀 NICEGOLD Enterprise - Single User Production Guide

## 📋 ภาพรวมระบบ

NICEGOLD Enterprise เป็นระบบ trading platform ที่ออกแบบมาเพื่อให้ผู้ดูแลระบบคนเดียวสามารถบริหารจัดการได้ทั้งหมด โดยใช้ AI เป็นทีมงานช่วยเหลือในการทำงานต่างๆ

### 🏗️ สถาปัตยกรรมระบบ

```
NICEGOLD Enterprise
├── 🔐 Single-User Authentication (PBKDF2 + JWT)
├── 🚀 FastAPI Backend (REST API)
├── 📊 Streamlit Dashboard (Web Interface)
├── 🤖 AI Team System (Virtual Agents)
├── 🎯 AI Orchestrator (Workflow Management)
├── 💾 SQLite Database (Production Da```

## ✅ Production Validation Checklist

### Pre-Deployment Validation

```bash
# 1. ตรวจสอบ dependencies
python final_production_validation.## 🔗 การจัดการ Git Repository

### การ Push ไปยัง GitHub

#### วิธีที่ 1: ใช้สคริปต์ Quick Push (แนะนำ)

```bash
# Push แบบง่ายๆ พร้อม commit message อัตโนมัติ
./quick_push.sh

# Push พร้อมกำหนด commit message เอง
./quick_push.sh "เพิ่มฟีเจอร์ใหม่สำหรับการวิเคราะห์ข้อมูล"
```

#### วิธีที่ 2: ใช้ Git Manager (สำหรับผู้ใช้ขั้นสูง)

```bash
# ตรวจสอบสถานะ repository
python git_manager.py --action status

# Commit และ push อย่างอัจฉริยะ
python git_manager.py --action smart-push

# Push พร้อมข้อความ
python git_manager.py --action smart-push --message "อัปเดตระบบ" --description "ปรับปรุงประสิทธิภาพ"

# สร้างรายงานสถานะ Git
python git_manager.py --action status --report
```

#### วิธีที่ 3: ใช้ Auto Deployment System

```bash
# Deploy แบบอัตโนมัติ (รวม backup, validation, และ push)
python auto_deployment.py

# Deploy พร้อมข้อความ commit
python auto_deployment.py --message "Release version 1.1.0"

# Deploy แบบ dry-run (ดูว่าจะทำอะไรบ้าง)
python auto_deployment.py --dry-run

# Deploy โดยไม่สร้าง backup
python auto_deployment.py --no-backup

# Deploy โดยไม่ push ไป Git
python auto_deployment.py --no-push
```

### การตั้งค่า Git Repository

```bash
# ตรวจสอบการตั้งค่า Git ปัจจุบัน
git config --list

# ตั้งค่า Git user (ถ้ายังไม่ได้ตั้ง)
git config user.name "NICEGOLD Administrator"
git config user.email "admin@nicegold.local"

# ตรวจสอบ remote repository
git remote -v

# ตรวจสอบสถานะ
git status

# ดู commit history
git log --oneline -10
```

### การแก้ปัญหาที่พบบ่อย

#### 1. ปัญหา Authentication

```bash
# ถ้าใช้ GitHub Personal Access Token
git remote set-url origin https://[TOKEN]@github.com/nicetpad2/NICEGOLD-ProjectP.git

# หรือใช้ SSH key
git remote set-url origin git@github.com:nicetpad2/NICEGOLD-ProjectP.git
```

#### 2. ปัญหา Merge Conflicts

```bash
# ดึงการเปลี่ยนแปลงล่าสุดจาก remote
git fetch origin

# Merge การเปลี่ยนแปลง
git merge origin/main

# หรือใช้ rebase
git rebase origin/main

# ถ้ามี conflict ให้แก้ไขแล้วทำ
git add .
git commit -m "Resolve merge conflicts"
```

#### 3. ปัญหา Large Files

```bash
# ใช้ Git LFS สำหรับไฟล์ขนาดใหญ่
git lfs install
git lfs track "*.csv"
git lfs track "*.parquet"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### การสำรองข้อมูลและ Restore

```bash
# สร้าง backup ก่อน push
python auto_deployment.py --no-push

# ดู backup ที่มี
ls -la backups/

# Restore จาก backup
cd backups/
tar -xzf nicegold_backup_YYYYMMDD_HHMMSS.tar.gz
# แล้วคัดลอกไฟล์ที่ต้องการกลับ
```

### การติดตาม Deployment

```bash
# ดูรายงาน deployment ล่าสุด
ls -la reports/deployment/

# ดู log การ deploy
tail -f logs/deployment/deployment_$(date +%Y%m%d).log

# ตรวจสอบสถานะ Git หลัง deploy
python git_manager.py --action status --report
```

### การตั้งค่า Automated Push

สร้างไฟล์ `config/deployment.yaml`:

```yaml
version: "1.0.0"
git:
  auto_push: true
  branch: "main"
  commit_prefix: "🚀 NICEGOLD"
  ignore_patterns:
    - "*.pyc"
    - "__pycache__"
    - "*.log"
    - "*.tmp"
    - ".env.local"

backup:
  enabled: true
  keep_backups: 5
  backup_dir: "backups"

validation:
  run_tests: true
  check_syntax: true
  check_dependencies: true

notifications:
  enabled: true
  on_success: true
  on_failure: true
```

### การสร้าง Release

```bash
# สร้าง tag สำหรับ release
git tag -a v1.0.0 -m "NICEGOLD Enterprise v1.0.0"
git push origin v1.0.0

# หรือใช้ auto deployment พร้อม tag
python auto_deployment.py --message "Release v1.0.0 - Production Ready"
git tag -a v1.0.0 -m "Production Release v1.0.0"
git push origin v1.0.0
```

## 📝 Changelog

### Version 1.1.0 (Latest)
- ✅ Git Repository Management System
- ✅ Auto Deployment with Backup
- ✅ Smart Push และ Commit System
- ✅ Repository Validation และ Testing
- ✅ Deployment Reports และ Notifications

### Version 1.0.0
- ✅ Single-user authentication system
- ✅ AI Team management
- ✅ AI Orchestrator
- ✅ Production deployment automation
- ✅ Real-time monitoring
- ✅ Automated backup system
- ✅ Comprehensive health checking

---

*สำหรับข้อมูลเพิ่มเติมและการอัปเดต โปรดตรวจสอบเอกสารล่าสุดใน [GitHub Repository](https://github.com/nicetpad2/NICEGOLD-ProjectP)*s

# 2. ตรวจสอบโครงสร้างข้อมูล
python -c "
import pandas as pd
import os

# ตรวจสอบไฟล์ข้อมูลหลัก
required_files = ['XAUUSD_M1.csv', 'XAUUSD_M15.csv']
for file in required_files:
    if os.path.exists(file):
        df = pd.read_csv(file, nrows=10)
        print(f'{file}: ✓ ({len(pd.read_csv(file))} rows)')
        print(f'  Columns: {list(df.columns)}')
    else:
        print(f'{file}: ✗ Missing')
"

# 3. ตรวจสอบการตั้งค่า
python -c "
import yaml
import os

configs = ['config.yaml', '.env.production', 'config/production.yaml']
for config in configs:
    if os.path.exists(config):
        print(f'{config}: ✓')
    else:
        print(f'{config}: ✗ Missing')
"

# 4. ตรวจสอบ permissions
ls -la production_single_user_deploy.py start_production_single_user.py
chmod +x *.py *.sh 2>/dev/null || true

# 5. ทดสอบ authentication
python -c "
from src.single_user_auth import SingleUserAuth
auth = SingleUserAuth()
print('Auth system: ✓ Ready')
"
```

### Post-Deployment Validation

```bash
# 1. ตรวจสอบสถานะบริการ
python system_status.py --full

# 2. ทดสอบ API endpoints
curl -X GET http://localhost:8000/health
curl -X GET http://localhost:8000/api/status

# 3. ตรวจสอบ logs
tail -f logs/production.log
tail -f logs/api.log
tail -f logs/system.log

# 4. ทดสอบ AI agents
python -c "
from ai_team_manager import AITeamManager
manager = AITeamManager()
print('AI Team Status:', manager.get_team_status())
"

# 5. ทดสอบ backup system
python system_maintenance.py --test-backup

# 6. ทดสอบ monitoring
python enhanced_production_monitor.py --test-mode
```

### Performance Validation

```bash
# 1. ทดสอบประสิทธิภาพระบบ
python -c "
import time
import psutil
import pandas as pd

print('System Performance Check:')
print(f'CPU Usage: {psutil.cpu_percent(interval=1)}%')
print(f'Memory Usage: {psutil.virtual_memory().percent}%')
print(f'Disk Usage: {psutil.disk_usage(\"/\").percent}%')

# ทดสอบการโหลดข้อมูล
start = time.time()
df = pd.read_csv('XAUUSD_M1.csv')
load_time = time.time() - start
print(f'Data Load Time: {load_time:.2f}s ({len(df)} rows)')
"

# 2. ทดสอบ API response time
python -c "
import requests
import time

try:
    start = time.time()
    response = requests.get('http://localhost:8000/health', timeout=5)
    response_time = (time.time() - start) * 1000
    print(f'API Response Time: {response_time:.0f}ms')
    print(f'API Status: {response.status_code}')
except Exception as e:
    print(f'API Test Failed: {e}')
"
```

### Security Validation

```bash
# 1. ตรวจสอบ file permissions
find . -name "*.py" -exec ls -la {} \; | head -10
find . -name "*.yaml" -exec ls -la {} \; | head -5
find . -name ".env*" -exec ls -la {} \; | head -5

# 2. ตรวจสอบ process security
ps aux | grep python | grep -E "(api|production)"

# 3. ตรวจสอบ network security
netstat -tlnp | grep -E "(8000|8080)"

# 4. ตรวจสอบ JWT security
python -c "
from src.single_user_auth import SingleUserAuth
auth = SingleUserAuth()
token = auth.create_test_token()
is_valid = auth.verify_token(token)
print(f'JWT Security Test: {\"✓ Pass\" if is_valid else \"✗ Fail\"}')
"
```

## 📝 Changelog
├── 📈 Monitoring & Logging
└── 🔒 Security & Backup Systems
```

### 🎯 ฟีเจอร์หลัก

- **Single-User Control**: ระบบควบคุมโดยผู้ดูแลคนเดียว
- **AI Team**: ทีม AI ที่ช่วยในการวิเคราะห์และตัดสินใจ
- **Production Ready**: พร้อมใช้งานจริงด้วยระบบรักษาความปลอดภัยที่แข็งแกร่ง
- **Real-time Monitoring**: ติดตามระบบแบบเรียลไทม์
- **Automated Backup**: ระบบสำรองข้อมูลอัตโนมัติ

## 🚀 Quick Start

### 1. การติดตั้งแบบ One-Click

```bash
# ติดตั้งและตั้งค่าระบบทั้งหมดในคำสั่งเดียว
python one_click_deploy.py
```

### 2. เริ่มใช้งานระบบ

```bash
# เริ่มบริการทั้งหมด
./start_services.sh

# หรือใช้ production manager
python start_production_single_user.py
```

### 3. เข้าถึงระบบ

- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🔧 การจัดการระบบ

### ระบบ Authentication

```bash
# จัดการผู้ใช้
python src/single_user_auth.py

# เปลี่ยนรหัสผ่าน
python src/single_user_auth.py --change-password

# ตรวจสอบสถานะการล็อกอิน
python src/single_user_auth.py --check-sessions
```

### ระบบ AI Team

```bash
# เปิด AI Team Manager
python ai_team_manager.py

# เปิด AI Orchestrator (ระบบจัดการ AI ทั้งหมด)
python ai_orchestrator.py

# ดู AI Assistant Brain
python ai_assistant_brain.py
```

### การตรวจสอบและบำรุงรักษา

```bash
# ตรวจสอบสุขภาพระบบ
python system_maintenance.py health

# เริ่มการติดตามระบบแบบเรียลไทม์
python system_maintenance.py monitor

# สำรองข้อมูล
python system_maintenance.py backup
```

### การทดสอบระบบ

```bash
# ทดสอบการทำงานของระบบทั้งหมด
python final_integration_live_test.py
```

## 🏗️ โครงสร้างไฟล์

```
NICEGOLD-ProjectP/
├── 📁 src/                     # Source code หลัก
│   ├── api.py                  # FastAPI backend
│   ├── single_user_auth.py     # ระบบ authentication
│   └── dashboard.py            # Streamlit dashboard
├── 📁 config/                  # ไฟล์ configuration
│   ├── production.yaml         # Config สำหรับ production
│   └── ai_orchestrator/        # Config สำหรับ AI
├── 📁 database/                # ฐานข้อมูล
│   └── production.db           # SQLite database
├── 📁 logs/                    # Log files
├── 📁 backups/                 # ไฟล์สำรองข้อมูล
├── 📁 run/                     # PID files
├── 🔧 one_click_deploy.py      # สคริปต์ติดตั้งแบบ one-click
├── 🚀 start_production_single_user.py  # เริ่มระบบ production
├── 🔧 system_maintenance.py    # ระบบบำรุงรักษา
├── 🤖 ai_team_manager.py       # จัดการทีม AI
├── 🎯 ai_orchestrator.py       # ประสานงาน AI
└── 🧠 ai_assistant_brain.py    # AI assistant
```

## 🔐 ระบบความปลอดภัย

### การตั้งค่าความปลอดภัย

1. **Single-User Authentication**
   - ใช้ PBKDF2 สำหรับการเข้ารหัสรหัสผ่าน
   - JWT tokens สำหรับ session management
   - Session timeout และ token expiration

2. **File Permissions**
   ```bash
   # ตรวจสอบการตั้งค่า permissions
   ls -la .env.production     # ควรเป็น 600 (owner only)
   ls -la database/           # ควรเป็น 700 (owner only)
   ```

3. **Network Security**
   - ใช้ localhost สำหรับการพัฒนา
   - สำหรับ production ภายนอก ควรใช้ SSL/TLS
   - ตั้งค่า firewall ให้เหมาะสม

### การสำรองข้อมูลและการกู้คืน

```bash
# สำรองข้อมูลแบบอัตโนมัติ (ทุก 1 ชั่วโมง)
python system_maintenance.py backup

# ดูไฟล์สำรองข้อมูล
ls -la backups/

# กู้คืนจากไฟล์สำรองข้อมูล
# 1. หยุดบริการทั้งหมด
./stop_services.sh

# 2. แตกไฟล์สำรองข้อมูล
cd backups/
tar -xzf nicegold_backup_YYYYMMDD_HHMMSS.tar.gz

# 3. คัดลอกไฟล์กลับ
cp backup_folder/production.db ../database/
cp backup_folder/config.yaml ../

# 4. เริ่มบริการใหม่
cd ..
./start_services.sh
```

## 🤖 คู่มือ AI Team

### AI Agents ที่มีอยู่

1. **📊 Data Analyst Agent**
   - วิเคราะห์ข้อมูลตลาด
   - สร้างรายงานและกราฟ
   - ตรวจสอบ data quality

2. **🎯 Strategy AI Agent**
   - พัฒนากลยุทธ์การเทรด
   - วิเคราะห์ความเสี่ยง
   - แนะนำการปรับปรุง

3. **⚡ Risk Manager Agent**
   - ประเมินความเสี่ยง
   - ตั้งค่า stop-loss
   - เตือนภัยในสถานการณ์อันตราย

4. **🔧 Technical Analyst Agent**
   - วิเคราะห์ทางเทคนิค
   - หาจังหวะเข้า-ออก
   - ดูแล indicators

5. **📈 Performance Monitor Agent**
   - ติดตามผลการดำเนินงาน
   - สร้างรายงานประสิทธิภาพ
   - เปรียบเทียบกับ benchmark

### การใช้งาน AI Team

```bash
# เปิด AI Team Dashboard
python ai_team_manager.py

# ส่งงานให้ AI Agent
python ai_team_manager.py --assign-task "วิเคราะห์ข้อมูลตลาดวันนี้" --agent "Data Analyst"

# ดูผลงานของ AI
python ai_team_manager.py --show-results

# ใช้ AI Orchestrator สำหรับงานที่ซับซ้อน
python ai_orchestrator.py
```

### AI Assistant Brain

```bash
# เริ่ม AI Assistant สำหรับการสนทนา
python ai_assistant_brain.py

# ใช้ในโหมด CLI
python ai_assistant_brain.py --cli

# ขอคำแนะนำ
python ai_assistant_brain.py --advice "ควรเทรดอะไรวันนี้"
```

## 📊 การติดตามและ Monitoring

### Dashboard การติดตามระบบ

```bash
# เปิด real-time monitoring
python system_maintenance.py monitor

# ตรวจสอบสุขภาพระบบ
python system_maintenance.py health
```

### การดู Logs

```bash
# API logs
tail -f logs/api/api_YYYYMMDD.log

# Dashboard logs  
tail -f logs/dashboard/dashboard_YYYYMMDD.log

# AI Team logs
tail -f logs/ai_team/ai_team_YYYYMMDD.log

# System logs
tail -f logs/maintenance/maintenance_YYYYMMDD.log
```

### การตรวจสอบประสิทธิภาพ

```bash
# ดู resource usage
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\".\").percent}%')
"

# ตรวจสอบ processes
ps aux | grep python
```

## 🔧 Troubleshooting

### ปัญหาที่พบบ่อย

#### 1. บริการไม่เริ่มต้น

```bash
# ตรวจสอบ port conflicts
netstat -tlnp | grep :8000
netstat -tlnp | grep :8501

# ตรวจสอบ PID files
ls -la run/

# หยุดบริการที่ค้างอยู่
./stop_services.sh

# เริ่มใหม่
./start_services.sh
```

#### 2. ปัญหาการเข้าสู่ระบบ

```bash
# ตรวจสอบ database
python src/single_user_auth.py --check-users

# รีเซ็ตรหัสผ่าน
python src/single_user_auth.py --reset-password admin

# ตรวจสอบ sessions
python src/single_user_auth.py --check-sessions
```

#### 3. ปัญหา AI Team

```bash
# ตรวจสอบ AI modules
python -c "
try:
    from ai_team_manager import AITeamManager
    print('AI Team Manager: OK')
except Exception as e:
    print(f'AI Team Manager Error: {e}')

try:
    from ai_orchestrator import AIOrchestrator  
    print('AI Orchestrator: OK')
except Exception as e:
    print(f'AI Orchestrator Error: {e}')
"

# รีสตาร์ท AI systems
python ai_orchestrator.py --reset
```

#### 4. ปัญหา Database

```bash
# ตรวจสอบ database integrity
sqlite3 database/production.db "PRAGMA integrity_check;"

# ดู tables
sqlite3 database/production.db ".tables"

# ตรวจสอบข้อมูล users
sqlite3 database/production.db "SELECT username, created_at, last_login FROM users;"
```

### การกู้คืนฉุกเฉิน

```bash
# 1. หยุดบริการทั้งหมด
./stop_services.sh

# 2. สำรองข้อมูลปัจจุบัน
cp -r database database_backup_$(date +%Y%m%d_%H%M%S)
cp -r config config_backup_$(date +%Y%m%d_%H%M%S)

# 3. กู้คืนจาก backup ล่าสุด
cd backups
latest_backup=$(ls -t *.tar.gz | head -n1)
tar -xzf $latest_backup
cp -r backup_folder/* ../

# 4. เริ่มระบบใหม่
cd ..
./start_services.sh

# 5. ตรวจสอบระบบ
python final_integration_live_test.py
```

## 🚀 การ Deploy Production

### การเตรียมสำหรับ Production

1. **อัปเดต Configuration**
   ```yaml
   # config/production.yaml
   app:
     environment: production
     debug: false
   
   server:
     host: 0.0.0.0  # สำหรับ external access
     ssl_enabled: true
   
   security:
     cors_enabled: true
     rate_limiting: true
   ```

2. **ตั้งค่า SSL/TLS**
   ```bash
   # สร้าง SSL certificates
   mkdir ssl
   openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes
   ```

3. **ตั้งค่า Firewall**
   ```bash
   # เปิดเฉพาะ ports ที่จำเป็น
   sudo ufw allow 8000/tcp  # API
   sudo ufw allow 8501/tcp  # Dashboard
   sudo ufw enable
   ```

4. **ตั้งค่า systemd services**
   ```bash
   # สร้าง service files
   sudo cp scripts/nicegold-api.service /etc/systemd/system/
   sudo cp scripts/nicegold-dashboard.service /etc/systemd/system/
   
   # เปิดใช้งาน
   sudo systemctl enable nicegold-api
   sudo systemctl enable nicegold-dashboard
   sudo systemctl start nicegold-api
   sudo systemctl start nicegold-dashboard
   ```

### การ Monitor Production

```bash
# ดูสถานะ services
sudo systemctl status nicegold-api
sudo systemctl status nicegold-dashboard

# ดู logs
journalctl -u nicegold-api -f
journalctl -u nicegold-dashboard -f

# ตรวจสอบ performance
python system_maintenance.py monitor --interval 10
```

## 📚 เอกสารเพิ่มเติม

- [API Documentation](http://localhost:8000/docs) - เอกสาร REST API
- [Configuration Guide](docs/configuration.md) - คู่มือการตั้งค่า
- [AI Team Guide](docs/ai_team.md) - คู่มือใช้งาน AI Team
- [Security Guide](docs/security.md) - คู่มือความปลอดภัย

## 🆘 การขอความช่วยเหลือ

### การรายงานปัญหา

1. **รวบรวมข้อมูล**
   ```bash
   # สร้าง diagnostic report
   python system_maintenance.py health > health_report.txt
   tar -czf diagnostic_$(date +%Y%m%d_%H%M%S).tar.gz logs/ health_report.txt
   ```

2. **ส่วนที่ต้องระบุ**
   - เวอร์ชันของระบบ
   - ข้อความ error ที่พบ
   - ขั้นตอนที่ทำก่อนเกิดปัญหา
   - Log files ที่เกี่ยวข้อง

### การติดต่อ

- GitHub Issues: [Repository Issues](https://github.com/your-repo/nicegold/issues)
- Documentation: [Wiki](https://github.com/your-repo/nicegold/wiki)

## � โครงสร้างข้อมูล XAUUSD

### ไฟล์ข้อมูลหลัก

ระบบใช้ข้อมูลราคาทองคำ (XAUUSD) ในรูปแบบ CSV สำหรับการวิเคราะห์และ backtesting:

#### XAUUSD_M1.csv (ข้อมูล 1 นาที)
```
Columns: ['Open', 'High', 'Low', 'Close', 'Volume', 'Time']
Format:
- Open, High, Low, Close: ราคาในรูปแบบ float
- Volume: ปริมาณการซื้อขายในรูปแบบ float
- Time: timestamp ในรูปแบบ "YYYY-MM-DD HH:MM:SS"

ตัวอย่างข้อมูล:
       Open      High       Low     Close   Volume                 Time
0  1726.685  1727.075  1726.495  1726.865  0.04444  2020-06-12 03:00:00
1  1726.895  1726.895  1725.305  1725.505  0.04070  2020-06-12 03:01:00
```

#### XAUUSD_M15.csv (ข้อมูล 15 นาที)
```
Columns: ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
Format:
- Timestamp: timestamp ในรูปแบบ "YYYY-MM-DD HH:MM:SS" 
- Open, High, Low, Close: ราคาในรูปแบบ float
- Volume: ปริมาณการซื้อขายในรูปแบบ float

ตัวอย่างข้อมูล:
             Timestamp      Open      High       Low     Close   Volume
0  2563-06-12 03:00:00  1726.685  1727.075  1722.135  1725.075  0.51033
1  2563-06-12 03:15:00  1725.065  1728.535  1724.935  1727.525  0.52918
```

### การใช้งานข้อมูล

```python
# โหลดข้อมูล M1
import pandas as pd
df_m1 = pd.read_csv('XAUUSD_M1.csv')
df_m1['Time'] = pd.to_datetime(df_m1['Time'])

# โหลดข้อมูล M15
df_m15 = pd.read_csv('XAUUSD_M15.csv')
df_m15['Timestamp'] = pd.to_datetime(df_m15['Timestamp'])

# ตรวจสอบข้อมูล
print(f"M1 Data: {len(df_m1)} rows, Period: {df_m1['Time'].min()} to {df_m1['Time'].max()}")
print(f"M15 Data: {len(df_m15)} rows, Period: {df_m15['Timestamp'].min()} to {df_m15['Timestamp'].max()}")
```

### การตรวจสอบคุณภาพข้อมูล

```bash
# ตรวจสอบโครงสร้างข้อมูล
python -c "
import pandas as pd
df1 = pd.read_csv('XAUUSD_M1.csv', nrows=5)
print('XAUUSD_M1.csv columns:', df1.columns.tolist())
df15 = pd.read_csv('XAUUSD_M15.csv', nrows=5)  
print('XAUUSD_M15.csv columns:', df15.columns.tolist())
"

# ตรวจสอบข้อมูลที่ขาดหาย
python -c "
import pandas as pd
df = pd.read_csv('XAUUSD_M1.csv')
print('Missing values in M1:', df.isnull().sum().sum())
df = pd.read_csv('XAUUSD_M15.csv')
print('Missing values in M15:', df.isnull().sum().sum())
"
```

## �📝 Changelog

### Version 1.0.0 (Latest)
- ✅ Single-user authentication system
- ✅ AI Team management
- ✅ AI Orchestrator
- ✅ Production deployment automation
- ✅ Real-time monitoring
- ✅ Automated backup system
- ✅ Comprehensive health checking

---

*สำหรับข้อมูลเพิ่มเติมและการอัปเดต โปรดตรวจสอบเอกสารล่าสุดในระบบ*
