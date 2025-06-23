#!/usr/bin/env python3
"""
Quick Migration Guide for ML Tracking System
คู่มือการย้ายโปรเจ็กต์และสภาพแวดล้อมแบบง่าย
"""

import os
import shutil
import zipfile
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

class QuickMigrator:
    """เครื่องมือย้ายโปรเจ็กต์แบบง่าย"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_export_package(self, output_name: str = None) -> str:
        """สร้าง export package สำหรับย้ายโปรเจ็กต์"""
        
        if output_name is None:
            output_name = f"ml_tracking_export_{self.timestamp}.zip"
        
        print("🚀 เริ่มสร้าง Export Package...")
        
        # รายการไฟล์และโฟลเดอร์ที่ต้องการ
        essential_items = [
            # Core tracking files
            "tracking.py",
            "tracking_config.yaml", 
            "tracking_cli.py",
            "tracking_integration.py",
            "tracking_examples.py",
            
            # Configuration files
            "requirements.txt",
            "tracking_requirements.txt",
            ".env.example",
            
            # Data and models
            "enterprise_tracking",
            "enterprise_mlruns", 
            "models",
            "artifacts",
            "data",
            
            # Project files
            "ProjectP.py",
            "config.yaml",
            "config",
            
            # Documentation
            "README_TRACKING.md",
            "SETUP_COMPLETE.md"
        ]
        
        with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # Add migration info
            migration_info = {
                'created': datetime.now().isoformat(),
                'source_path': str(self.project_path),
                'platform': os.name,
                'included_items': [],
                'instructions': {
                    'setup': [
                        '1. แตก zip file ในตำแหน่งใหม่',
                        '2. ติดตั้ง dependencies: pip install -r tracking_requirements.txt',
                        '3. Copy .env.example เป็น .env และตั้งค่า',
                        '4. รัน: python enterprise_setup_tracking.py',
                        '5. ทดสอบ: python tracking_examples.py'
                    ],
                    'validation': [
                        'ตรวจสอบ MLflow UI: mlflow ui --port 5000',
                        'ตรวจสอบ tracking: python tracking_cli.py status',
                        'ทดสอบ dashboard: streamlit run dashboard_app.py'
                    ]
                }
            }
            
            # Add each item to zip
            added_count = 0
            for item in essential_items:
                item_path = self.project_path / item
                
                if item_path.exists():
                    if item_path.is_file():
                        zipf.write(item_path, item)
                        migration_info['included_items'].append(f"📄 {item}")
                        added_count += 1
                        print(f"  ✅ เพิ่มไฟล์: {item}")
                    
                    elif item_path.is_dir():
                        for file_path in item_path.rglob('*'):
                            if file_path.is_file():
                                rel_path = file_path.relative_to(self.project_path)
                                zipf.write(file_path, rel_path)
                        migration_info['included_items'].append(f"📁 {item}/")
                        added_count += 1
                        print(f"  ✅ เพิ่มโฟลเดอร์: {item}/")
                else:
                    print(f"  ⚠️ ไม่พบ: {item}")
            
            # Add migration info to zip
            zipf.writestr('MIGRATION_INFO.json', json.dumps(migration_info, indent=2, ensure_ascii=False))
            
            # Create setup script for new environment
            setup_script = self._create_setup_script()
            zipf.writestr('setup_new_environment.py', setup_script)
            
            # Create quick start guide
            quick_guide = self._create_quick_guide()
            zipf.writestr('QUICK_START.md', quick_guide)
        
        print(f"✅ Export Package สร้างเสร็จแล้ว: {output_name}")
        print(f"📦 รวมไฟล์/โฟลเดอร์: {added_count} รายการ")
        print(f"💾 ขนาดไฟล์: {os.path.getsize(output_name) / (1024*1024):.1f} MB")
        
        return output_name
    
    def import_package(self, zip_file: str, target_path: str = None) -> bool:
        """Import package ในสภาพแวดล้อมใหม่"""
        
        if target_path is None:
            target_path = f"imported_project_{self.timestamp}"
        
        target_path = Path(target_path).resolve()
        target_path.mkdir(exist_ok=True)
        
        print(f"📥 กำลัง Import จาก: {zip_file}")
        print(f"📁 ไปยัง: {target_path}")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zipf:
                zipf.extractall(target_path)
            
            print("✅ แตกไฟล์เสร็จแล้ว")
            
            # Show next steps
            migration_info_path = target_path / 'MIGRATION_INFO.json'
            if migration_info_path.exists():
                with open(migration_info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                print("\\n📋 ขั้นตอนต่อไป:")
                for i, step in enumerate(info['instructions']['setup'], 1):
                    print(f"  {i}. {step}")
                
                print("\\n🔍 การตรวจสอบ:")
                for i, check in enumerate(info['instructions']['validation'], 1):
                    print(f"  {i}. {check}")
            
            return True
            
        except Exception as e:
            print(f"❌ Import ล้มเหลว: {e}")
            return False
    
    def _create_setup_script(self) -> str:
        """สร้างสคริปต์สำหรับติดตั้งในสภาพแวดล้อมใหม่"""
        
        return '''#!/usr/bin/env python3
"""
Setup Script for New Environment
สคริปต์ติดตั้งระบบในสภาพแวดล้อมใหม่
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 กำลังติดตั้งระบบ ML Tracking ในสภาพแวดล้อมใหม่...")
    
    # ตรวจสอบ Python version
    if sys.version_info < (3, 8):
        print("❌ ต้องการ Python 3.8 ขึ้นไป")
        return False
    
    print(f"✅ Python {sys.version}")
    
    # ติดตั้ง dependencies
    print("📦 ติดตั้ง dependencies...")
    
    requirements_files = [
        "tracking_requirements.txt",
        "requirements.txt"
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], 
                              check=True)
                print(f"✅ ติดตั้งจาก {req_file} เสร็จแล้ว")
            except subprocess.CalledProcessError:
                print(f"⚠️ ติดตั้งจาก {req_file} มีปัญหา")
    
    # สร้างโฟลเดอร์ที่จำเป็น
    print("📁 สร้างโฟลเดอร์...")
    folders = [
        "enterprise_tracking", "enterprise_mlruns", "models", 
        "artifacts", "logs", "data", "reports", "backups"
    ]
    
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
        print(f"  ✅ {folder}/")
    
    # ตั้งค่า environment
    print("⚙️ ตั้งค่า environment...")
    
    if Path(".env.example").exists() and not Path(".env").exists():
        import shutil
        shutil.copy(".env.example", ".env")
        print("  ✅ สร้าง .env จาก .env.example")
        print("  ⚠️ กรุณาแก้ไข .env ตามสภาพแวดล้อมของคุณ")
    
    # ทดสอบระบบ
    print("🧪 ทดสอบระบบ...")
    
    try:
        # Test MLflow import
        import mlflow
        print("  ✅ MLflow พร้อมใช้งาน")
        
        # Test tracking system
        if Path("tracking.py").exists():
            from tracking import ExperimentTracker
            tracker = ExperimentTracker()
            print("  ✅ Tracking system พร้อมใช้งาน")
        
    except ImportError as e:
        print(f"  ⚠️ ปัญหาการ import: {e}")
    
    print("\\n🎉 การติดตั้งเสร็จสิ้น!")
    print("\\n📋 ขั้นตอนต่อไป:")
    print("1. แก้ไขไฟล์ .env")
    print("2. รัน: python tracking_examples.py")
    print("3. เปิด MLflow UI: mlflow ui --port 5000")
    print("4. ทดสอบ CLI: python tracking_cli.py status")
    
    return True

if __name__ == "__main__":
    main()
'''
    
    def _create_quick_guide(self) -> str:
        """สร้างคู่มือเริ่มต้นอย่างเร็ว"""
        
        return '''# 🚀 Quick Start Guide - ML Tracking System

## 📦 การติดตั้งใหม่

### 1️⃣ ขั้นตอนพื้นฐาน

```bash
# 1. แตก zip file และเข้าไปในโฟลเดอร์
cd imported_project_folder

# 2. รันสคริปต์ติดตั้ง
python setup_new_environment.py

# 3. ทดสอบระบบ
python tracking_examples.py
```

### 2️⃣ การตั้งค่า Environment Variables

แก้ไขไฟล์ `.env`:

```env
# MLflow
MLFLOW_TRACKING_URI=./enterprise_mlruns

# Weights & Biases (ถ้าใช้)
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=your_project_name

# Database (ถ้าใช้)
DATABASE_URL=your_database_url

# Monitoring
ENABLE_MONITORING=true
ALERT_EMAIL=your_email@domain.com
```

### 3️⃣ การทดสอบระบบ

```bash
# เปิด MLflow UI
mlflow ui --backend-store-uri ./enterprise_mlruns --port 5000

# ทดสอบ tracking
python -c "from tracking import ExperimentTracker; print('✅ OK')"

# ทดสอบ CLI
python tracking_cli.py status

# เริ่ม dashboard
streamlit run dashboard_app.py
```

## 🔗 การเข้าถึงบริการ

- **MLflow UI**: http://localhost:5000
- **Dashboard**: http://localhost:8501
- **Monitoring**: http://localhost:8502

## 🆘 หากมีปัญหา

### ปัญหาทั่วไป:

1. **Import Error**: ติดตั้ง dependencies ใหม่
   ```bash
   pip install -r tracking_requirements.txt
   ```

2. **Permission Error**: ตรวจสอบสิทธิ์ไฟล์
   ```bash
   chmod -R 755 enterprise_tracking/
   ```

3. **MLflow Connection Error**: ตรวจสอบ tracking URI
   ```bash
   export MLFLOW_TRACKING_URI=./enterprise_mlruns
   ```

### การขอความช่วยเหลือ:

1. ดูไฟล์ `README_TRACKING.md` สำหรับคู่มือละเอียด
2. ตรวจสอบ logs ในโฟลเดอร์ `logs/`
3. รัน health check: `python tracking_cli.py health-check`

## 🎯 การใช้งานเบื้องต้น

```python
from tracking import ExperimentTracker

# เริ่ม experiment
tracker = ExperimentTracker()

with tracker.start_run("my_first_experiment") as run:
    # ฝึก model
    model = train_your_model()
    
    # Log metrics
    run.log_metric("accuracy", 0.95)
    
    # Log model
    run.log_model(model, "my_model")
```

## 🏭 Production Deployment

สำหรับ production:

1. ใช้ database backend แทน file-based
2. ตั้งค่า cloud storage สำหรับ artifacts
3. เปิด monitoring และ alerting
4. ใช้ Docker หรือ Kubernetes

ดูรายละเอียดใน `README_TRACKING.md` section "Production Deployment"

---

**🎉 ขอให้ใช้งานอย่างมีความสุข!**
'''

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Migration Tool for ML Tracking System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Create export package')
    export_parser.add_argument('--output', '-o', help='Output file name')
    
    # Import command  
    import_parser = subparsers.add_parser('import', help='Import package')
    import_parser.add_argument('--input', '-i', required=True, help='Input zip file')
    import_parser.add_argument('--target', '-t', help='Target directory')
    
    args = parser.parse_args()
    
    migrator = QuickMigrator()
    
    if args.command == 'export':
        output_file = migrator.create_export_package(args.output)
        print(f"\\n📋 ขั้นตอนต่อไป:")
        print(f"1. ส่ง {output_file} ไปยังสภาพแวดล้อมใหม่")
        print(f"2. รัน: python quick_migration.py import -i {output_file}")
        
    elif args.command == 'import':
        success = migrator.import_package(args.input, args.target)
        if success:
            print("\\n✅ Import เสร็จสิ้น!")
            print("📋 รันสคริปต์ติดตั้ง: python setup_new_environment.py")
        else:
            print("\\n❌ Import ล้มเหลว")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
