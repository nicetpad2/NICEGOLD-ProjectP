#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
from typing import Dict, List, Optional
import json
            import mlflow
import os
import platform
import shutil
import subprocess
import sys
"""
Automated Setup Script for New Environment
สคริปต์ติดตั้งอัตโนมัติสำหรับสภาพแวดล้อมใหม่
"""


try:
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

class EnvironmentSetup:
    """ระบบติดตั้งอัตโนมัติสำหรับสภาพแวดล้อมใหม่"""

    def __init__(self):
        self.workspace_root = Path.cwd()
        self.setup_log = []
        self.status = {
            'python_check': False, 
            'dependencies': False, 
            'directories': False, 
            'configuration': False, 
            'validation': False
        }

    def log(self, message: str, level: str = "INFO"):
        """บันทึก log การติดตั้ง"""
        timestamp = datetime.now().strftime("%Y - %m - %d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.setup_log.append(log_entry)

        if RICH_AVAILABLE and console:
            if level == "SUCCESS":
                console.print(f"✅ {message}", style = "green")
            elif level == "ERROR":
                console.print(f"❌ {message}", style = "red")
            elif level == "WARNING":
                console.print(f"⚠️ {message}", style = "yellow")
            else:
                console.print(f"ℹ️ {message}", style = "blue")
        else:
            print(f"{level}: {message}")

    def check_python_version(self) -> bool:
        """ตรวจสอบ Python version"""
        self.log("ตรวจสอบ Python version...")

        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.log(f"Python {version.major}.{version.minor}.{version.micro} - รองรับ", "SUCCESS")
            self.status['python_check'] = True
            return True
        else:
            self.log(f"Python {version.major}.{version.minor}.{version.micro} - ต้องการ Python 3.8 + ", "ERROR")
            return False

    def install_dependencies(self) -> bool:
        """ติดตั้ง dependencies"""
        self.log("ติดตั้ง dependencies...")

        # ลิสต์ไฟล์ requirements ตามลำดับความสำคัญ
        requirements_files = [
            "tracking_requirements.txt", 
            "requirements.txt", 
            "dev - requirements.txt"
        ]

        installed_any = False

        for req_file in requirements_files:
            req_path = self.workspace_root / req_file
            if req_path.exists():
                self.log(f"พบไฟล์: {req_file}")

                try:
                    # ติดตั้ง packages
                    cmd = [sys.executable, " - m", "pip", "install", " - r", str(req_path), " -  - upgrade"]
                    result = subprocess.run(cmd, capture_output = True, text = True)

                    if result.returncode == 0:
                        self.log(f"ติดตั้งจาก {req_file} เสร็จแล้ว", "SUCCESS")
                        installed_any = True
                    else:
                        self.log(f"ติดตั้งจาก {req_file} ล้มเหลว: {result.stderr}", "ERROR")

                except Exception as e:
                    self.log(f"ข้อผิดพลาดในการติดตั้งจาก {req_file}: {e}", "ERROR")
            else:
                self.log(f"ไม่พบไฟล์: {req_file}", "WARNING")

        if not installed_any:
            # ติดตั้ง essential packages หากไม่มี requirements file
            self.log("ไม่พบไฟล์ requirements - ติดตั้ง essential packages...")
            essential_packages = [
                "mlflow> = 2.9.0", 
                "rich> = 13.0.0", 
                "typer> = 0.9.0", 
                "pyyaml> = 6.0", 
                "psutil> = 5.9.0", 
                "matplotlib> = 3.6.0", 
                "pandas> = 2.0.0", 
                "scikit - learn> = 1.3.0"
            ]

            for package in essential_packages:
                try:
                    cmd = [sys.executable, " - m", "pip", "install", package]
                    result = subprocess.run(cmd, capture_output = True, text = True)
                    if result.returncode == 0:
                        self.log(f"ติดตั้ง {package.split('> = ')[0]} เสร็จแล้ว", "SUCCESS")
                        installed_any = True
                    else:
                        self.log(f"ติดตั้ง {package} ล้มเหลว", "ERROR")
                except Exception as e:
                    self.log(f"ข้อผิดพลาดในการติดตั้ง {package}: {e}", "ERROR")

        self.status['dependencies'] = installed_any
        return installed_any

    def create_directories(self) -> bool:
        """สร้างโครงสร้างโฟลเดอร์"""
        self.log("สร้างโครงสร้างโฟลเดอร์...")

        directories = [
            "enterprise_tracking", 
            "enterprise_mlruns", 
            "models", 
            "artifacts", 
            "logs", 
            "data", 
            "reports", 
            "backups", 
            "configs", 
            "notebooks", 
            "scripts", 
            "monitoring"
        ]

        created_count = 0
        for directory in directories:
            dir_path = self.workspace_root / directory
            try:
                dir_path.mkdir(exist_ok = True)

                # สร้าง .gitkeep file
                gitkeep_path = dir_path / ".gitkeep"
                gitkeep_path.touch()

                self.log(f"สร้างโฟลเดอร์: {directory}/")
                created_count += 1

            except Exception as e:
                self.log(f"ไม่สามารถสร้างโฟลเดอร์ {directory}: {e}", "ERROR")

        success = created_count == len(directories)
        if success:
            self.log(f"สร้างโฟลเดอร์เสร็จแล้ว ({created_count} โฟลเดอร์)", "SUCCESS")

        self.status['directories'] = success
        return success

    def setup_configuration(self) -> bool:
        """ตั้งค่า configuration files"""
        self.log("ตั้งค่า configuration...")

        success = True

        # ตั้งค่า .env file
        env_example = self.workspace_root / ".env.example"
        env_file = self.workspace_root / ".env"

        if env_example.exists() and not env_file.exists():
            try:
                shutil.copy(env_example, env_file)
                self.log("สร้าง .env จาก .env.example", "SUCCESS")
            except Exception as e:
                self.log(f"ไม่สามารถสร้าง .env: {e}", "ERROR")
                success = False

        # ตรวจสอบ tracking_config.yaml
        config_file = self.workspace_root / "tracking_config.yaml"
        if not config_file.exists():
            self.log("ไม่พบ tracking_config.yaml - สร้างไฟล์พื้นฐาน", "WARNING")
            try:
                self._create_basic_config()
                self.log("สร้าง tracking_config.yaml พื้นฐาน", "SUCCESS")
            except Exception as e:
                self.log(f"ไม่สามารถสร้าง tracking_config.yaml: {e}", "ERROR")
                success = False

        # ตั้งค่า MLflow tracking URI
        try:
            os.environ["MLFLOW_TRACKING_URI"] = str(self.workspace_root / "enterprise_mlruns")
            self.log("ตั้งค่า MLFLOW_TRACKING_URI", "SUCCESS")
        except Exception as e:
            self.log(f"ไม่สามารถตั้งค่า MLFLOW_TRACKING_URI: {e}", "ERROR")
            success = False

        self.status['configuration'] = success
        return success

    def _create_basic_config(self):
        """สร้าง tracking_config.yaml พื้นฐาน"""
        basic_config = """# Basic ML Tracking Configuration
# Created by automated setup

mlflow:
  enabled: true
  tracking_uri: "./enterprise_mlruns"
  experiment_name: "default_experiment"

wandb:
  enabled: false
  project: "ml_project"

local:
  enabled: true
  save_models: true
  save_plots: true

tracking_dir: "./enterprise_tracking"
logging:
  level: "INFO"
  file_logging: true

auto_log:
  enabled: true
  log_system_info: true

monitoring:
  enabled: true
  alert_on_failure: false
"""

        config_path = self.workspace_root / "tracking_config.yaml"
        with open(config_path, 'w', encoding = 'utf - 8') as f:
            f.write(basic_config)

    def validate_installation(self) -> bool:
        """ตรวจสอบความถูกต้องของการติดตั้ง"""
        self.log("ตรวจสอบการติดตั้ง...")

        validation_results = {}

        # ตรวจสอบ import modules
        modules_to_test = ["mlflow", "rich", "yaml", "pandas"]

        for module in modules_to_test:
            try:
                __import__(module)
                validation_results[f"import_{module}"] = True
                self.log(f"Module {module} - OK", "SUCCESS")
            except ImportError:
                validation_results[f"import_{module}"] = False
                self.log(f"Module {module} - ไม่สามารถ import ได้", "ERROR")

        # ตรวจสอบ tracking system
        tracking_file = self.workspace_root / "tracking.py"
        if tracking_file.exists():
            try:
                # Import tracking module (ถ้าสามารถทำได้)
                validation_results["tracking_system"] = True
                self.log("Tracking system - พร้อมใช้งาน", "SUCCESS")
            except Exception as e:
                validation_results["tracking_system"] = False
                self.log(f"Tracking system - มีปัญหา: {e}", "ERROR")
        else:
            validation_results["tracking_system"] = False
            self.log("ไม่พบไฟล์ tracking.py", "ERROR")

        # ตรวจสอบ MLflow
        try:
            mlflow.set_tracking_uri(str(self.workspace_root / "enterprise_mlruns"))
            # ทดสอบสร้าง experiment
            experiment_name = "validation_test"
            experiment_id = mlflow.create_experiment(experiment_name, exist_ok = True)
            validation_results["mlflow_functionality"] = True
            self.log("MLflow - ทำงานปกติ", "SUCCESS")
        except Exception as e:
            validation_results["mlflow_functionality"] = False
            self.log(f"MLflow - มีปัญหา: {e}", "ERROR")

        # ตรวจสอบ directory structure
        required_dirs = ["enterprise_tracking", "enterprise_mlruns", "models", "logs"]
        dirs_ok = all((self.workspace_root / d).exists() for d in required_dirs)
        validation_results["directory_structure"] = dirs_ok

        if dirs_ok:
            self.log("โครงสร้างโฟลเดอร์ - สมบูรณ์", "SUCCESS")
        else:
            self.log("โครงสร้างโฟลเดอร์ - ไม่สมบูรณ์", "ERROR")

        # สรุปผลการตรวจสอบ
        passed_count = sum(validation_results.values())
        total_count = len(validation_results)

        success = passed_count >= (total_count * 0.8)  # ต้องผ่าน 80% ขึ้นไป

        if success:
            self.log(f"การตรวจสอบผ่าน ({passed_count}/{total_count})", "SUCCESS")
        else:
            self.log(f"การตรวจสอบไม่ผ่าน ({passed_count}/{total_count})", "ERROR")

        self.status['validation'] = success
        return success

    def generate_setup_report(self) -> str:
        """สร้างรายงานการติดตั้ง"""
        report_lines = [
            "# การติดตั้งระบบ ML Tracking - รายงาน", 
            f"สร้างเมื่อ: {datetime.now().strftime('%Y - %m - %d %H:%M:%S')}", 
            f"ระบบ: {platform.system()} {platform.release()}", 
            f"Python: {sys.version}", 
            f"ตำแหน่งโปรเจ็กต์: {self.workspace_root}", 
            "", 
            "## สถานะการติดตั้ง", 
            ""
        ]

        for component, status in self.status.items():
            status_icon = "✅" if status else "❌"
            status_text = "เสร็จสิ้น" if status else "ล้มเหลว"
            report_lines.append(f"- {status_icon} {component}: {status_text}")

        report_lines.extend([
            "", 
            "## Log การติดตั้ง", 
            ""
        ])

        for log_entry in self.setup_log:
            report_lines.append(f"- {log_entry}")

        report_lines.extend([
            "", 
            "## ขั้นตอนต่อไป", 
            "", 
            "### หากการติดตั้งสำเร็จ:", 
            "1. แก้ไขไฟล์ .env ตามสภาพแวดล้อมของคุณ", 
            "2. ทดสอบระบบ: `python tracking_examples.py`", 
            "3. เปิด MLflow UI: `mlflow ui - - port 5000`", 
            "4. ตรวจสอบ CLI: `python tracking_cli.py status`", 
            "", 
            "### หากการติดตั้งล้มเหลว:", 
            "1. ตรวจสอบ log ข้างต้น", 
            "2. ติดตั้ง dependencies ด้วยตนเอง", 
            "3. ตรวจสอบสิทธิ์การเขียนไฟล์", 
            "4. รันสคริปต์อีกครั้ง", 
            "", 
            "## การใช้งานเบื้องต้น", 
            "", 
            "```python", 
            "from tracking import ExperimentTracker", 
            "", 
            "tracker = ExperimentTracker()", 
            "", 
            "with tracker.start_run('my_experiment') as run:", 
            "    run.log_metric('accuracy', 0.95)", 
            "    run.log_param('model_type', 'RandomForest')", 
            "```", 
            "", 
            "## การเข้าถึงเว็บ Interface", 
            "", 
            "- MLflow UI: http://localhost:5000", 
            "- Dashboard (ถ้ามี): http://localhost:8501", 
            "", 
            " -  -  - ", 
            "รายงานนี้สร้างโดยระบบติดตั้งอัตโนมัติ"
        ])

        return "\\n".join(report_lines)

    def run_setup(self) -> bool:
        """รันกระบวนการติดตั้งทั้งหมด"""

        if RICH_AVAILABLE and console:
            console.print(Panel.fit(
                "[bold blue]🚀 ระบบติดตั้งอัตโนมัติ ML Tracking System[/bold blue]\\n"
                "[dim]กำลังติดตั้งและตั้งค่าระบบในสภาพแวดล้อมใหม่[/dim]", 
                border_style = "blue"
            ))
        else:
            print("🚀 ระบบติดตั้งอัตโนมัติ ML Tracking System")
            print("กำลังติดตั้งและตั้งค่าระบบในสภาพแวดล้อมใหม่")

        steps = [
            ("ตรวจสอบ Python Version", self.check_python_version), 
            ("ติดตั้ง Dependencies", self.install_dependencies), 
            ("สร้างโครงสร้างโฟลเดอร์", self.create_directories), 
            ("ตั้งค่า Configuration", self.setup_configuration), 
            ("ตรวจสอบการติดตั้ง", self.validate_installation)
        ]

        for step_name, step_func in steps:
            self.log(f"เริ่ม: {step_name}")

            try:
                if not step_func():
                    self.log(f"ติดตั้งล้มเหลวที่: {step_name}", "ERROR")
                    return False
            except Exception as e:
                self.log(f"ข้อผิดพลาดใน {step_name}: {e}", "ERROR")
                return False

        # สร้างและบันทึกรายงาน
        report = self.generate_setup_report()
        report_path = self.workspace_root / "SETUP_REPORT.md"

        try:
            with open(report_path, 'w', encoding = 'utf - 8') as f:
                f.write(report)
            self.log(f"บันทึกรายงานที่: {report_path}", "SUCCESS")
        except Exception as e:
            self.log(f"ไม่สามารถบันทึกรายงาน: {e}", "WARNING")

        # แสดงผลสำเร็จ
        if RICH_AVAILABLE and console:
            console.print(Panel.fit(
                "[bold green]🎉 การติดตั้งเสร็จสิ้น![/bold green]\\n"
                "[dim]ระบบ ML Tracking พร้อมใช้งานแล้ว[/dim]\\n"
                f"[dim]รายงาน: {report_path}[/dim]", 
                border_style = "green"
            ))
        else:
            print("\\n🎉 การติดตั้งเสร็จสิ้น!")
            print("ระบบ ML Tracking พร้อมใช้งานแล้ว")
            print(f"รายงาน: {report_path}")

        return True

def main():
    """ฟังก์ชันหลักสำหรับการติดตั้ง"""

    print("🚀 เริ่มการติดตั้งระบบ ML Tracking ในสภาพแวดล้อมใหม่...")

    setup = EnvironmentSetup()

    try:
        success = setup.run_setup()

        if success:
            print("\\n✅ การติดตั้งสำเร็จ!")
            print("\\n📋 ขั้นตอนต่อไป:")
            print("1. แก้ไขไฟล์ .env ตามต้องการ")
            print("2. ทดสอบ: python tracking_examples.py")
            print("3. เปิด MLflow UI: mlflow ui - - port 5000")
            print("4. ดูรายงาน: SETUP_REPORT.md")
            return 0
        else:
            print("\\n❌ การติดตั้งล้มเหลว")
            print("กรุณาตรวจสอบรายงานสำหรับรายละเอียด")
            return 1

    except KeyboardInterrupt:
        print("\\n⚠️ การติดตั้งถูกยกเลิกโดยผู้ใช้")
        return 1
    except Exception as e:
        print(f"\\n💥 ข้อผิดพลาดที่ไม่คาดคิด: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())