# -*- coding: utf - 8 -* -
#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

"""
🚀 NICEGOLD Auto Deployment System
ระบบ deploy อัตโนมัติสำหรับ NICEGOLD Enterprise
"""


class NiceGoldDeployment:
    """ระบบ deployment สำหรับ NICEGOLD"""

    def __init__(self, repo_path: str = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.setup_logging()
        self.deployment_config = self.load_deployment_config()

    def setup_logging(self):
        """ตั้งค่า logging"""
        log_dir = self.repo_path / "logs" / "deployment"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"deployment_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf - 8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def load_deployment_config(self) -> Dict:
        """โหลดการตั้งค่า deployment"""
        config_path = self.repo_path / "config" / "deployment.yaml"

        default_config = {
            "version": "1.0.0",
            "git": {
                "auto_push": True,
                "branch": "main",
                "commit_prefix": "🚀 NICEGOLD",
                "ignore_patterns": [
                    "*.pyc",
                    "__pycache__",
                    "*.log",
                    "*.tmp",
                    ".env.local",
                    "node_modules",
                    ".DS_Store",
                ],
            },
            "backup": {"enabled": True, "keep_backups": 5, "backup_dir": "backups"},
            "validation": {
                "run_tests": True,
                "check_syntax": True,
                "check_dependencies": True,
            },
            "notifications": {"enabled": True, "on_success": True, "on_failure": True},
        }

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf - 8") as f:
                    loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load config file: {e}")

        return default_config

    def save_deployment_config(self):
        """บันทึกการตั้งค่า deployment"""
        config_dir = self.repo_path / "config"
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "deployment.yaml"

        try:
            with open(config_path, "w", encoding="utf - 8") as f:
                yaml.dump(
                    self.deployment_config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                )
            self.logger.info(f"✅ Deployment config saved: {config_path}")
        except ImportError:
            # Fallback to JSON if PyYAML not available
            config_path = config_dir / "deployment.json"
            with open(config_path, "w", encoding="utf - 8") as f:
                json.dump(self.deployment_config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✅ Deployment config saved as JSON: {config_path}")

    def run_command(self, command: List[str], cwd: Path = None) -> tuple[bool, str]:
        """เรียกใช้คำสั่ง shell"""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            success = result.returncode == 0
            output = result.stdout if success else result.stderr

            if success:
                self.logger.debug(f"✅ Command succeeded: {' '.join(command)}")
            else:
                self.logger.error(f"❌ Command failed: {' '.join(command)}")
                self.logger.error(f"Error: {output}")

            return success, output.strip()

        except subprocess.TimeoutExpired:
            self.logger.error(f"⏰ Command timed out: {' '.join(command)}")
            return False, "Command timed out"
        except Exception as e:
            self.logger.error(f"💥 Command error: {e}")
            return False, str(e)

    def cleanup_repository(self):
        """ทำความสะอาด repository"""
        self.logger.info("🧹 Cleaning up repository...")

        ignore_patterns = self.deployment_config["git"]["ignore_patterns"]

        for pattern in ignore_patterns:
            if pattern.startswith("*."):
                # ลบไฟล์ตาม extension
                extension = pattern[1:]  # ลบ * ออก
                for file in self.repo_path.glob(f"**/*{extension}"):
                    try:
                        file.unlink()
                        self.logger.debug(f"Deleted: {file}")
                    except Exception as e:
                        self.logger.warning(f"Could not delete {file}: {e}")
            else:
                # ลบโฟลเดอร์หรือไฟล์
                for item in self.repo_path.glob(f"**/{pattern}"):
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        self.logger.debug(f"Deleted: {item}")
                    except Exception as e:
                        self.logger.warning(f"Could not delete {item}: {e}")

        self.logger.info("✅ Repository cleanup completed")

    def validate_deployment(self) -> bool:
        """ตรวจสอบความถูกต้องก่อน deployment"""
        self.logger.info("🔍 Validating deployment...")

        validation_config = self.deployment_config["validation"]

        # ตรวจสอบ syntax
        if validation_config.get("check_syntax", True):
            if not self.check_python_syntax():
                return False

        # ตรวจสอบ dependencies
        if validation_config.get("check_dependencies", True):
            if not self.check_dependencies():
                return False

        # รัน tests
        if validation_config.get("run_tests", True):
            if not self.run_tests():
                self.logger.warning("⚠️ Tests failed, but continuing deployment...")

        self.logger.info("✅ Validation completed")
        return True

    def check_python_syntax(self) -> bool:
        """ตรวจสอบ Python syntax"""
        self.logger.info("🐍 Checking Python syntax...")

        python_files = list(self.repo_path.glob("**/*.py"))
        errors = []

        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf - 8") as f:
                    compile(f.read(), py_file, "exec")
            except SyntaxError as e:
                errors.append(f"{py_file}: {e}")
            except Exception as e:
                self.logger.warning(f"Could not check {py_file}: {e}")

        if errors:
            self.logger.error("❌ Python syntax errors found:")
            for error in errors:
                self.logger.error(f"  {error}")
            return False

        self.logger.info(
            f"✅ Checked {len(python_files)} Python files - no syntax errors"
        )
        return True

    def check_dependencies(self) -> bool:
        """ตรวจสอบ dependencies"""
        self.logger.info("📦 Checking dependencies...")

        requirements_file = self.repo_path / "requirements.txt"

        if not requirements_file.exists():
            self.logger.warning("⚠️ requirements.txt not found")
            return True

        # ตรวจสอบว่า pip install สามารถทำได้
        success, output = self.run_command([sys.executable, " - m", "pip", "check"])

        if not success:
            self.logger.error(f"❌ Dependency check failed: {output}")
            return False

        self.logger.info("✅ Dependencies check passed")
        return True

    def run_tests(self) -> bool:
        """รัน unit tests"""
        self.logger.info("🧪 Running tests...")

        # หา test files
        test_files = (
            list(self.repo_path.glob("test_*.py"))
            + list(self.repo_path.glob("tests/**/*.py"))
            + list(self.repo_path.glob("**/test_*.py"))
        )

        if not test_files:
            self.logger.warning("⚠️ No test files found")
            return True

        # รัน pytest ถ้ามี
        try:
            success, output = self.run_command(
                [sys.executable, " - m", "pytest", " - v", " -  - tb = short"]
            )

            if success:
                self.logger.info("✅ All tests passed")
            else:
                self.logger.warning(f"⚠️ Some tests failed: {output}")

            return success

        except FileNotFoundError:
            # Fallback ใช้ unittest
            self.logger.info("pytest not found, using unittest...")

            for test_file in test_files:
                success, output = self.run_command([sys.executable, str(test_file)])

                if not success:
                    self.logger.warning(f"⚠️ Test failed: {test_file}")

            return True

    def create_backup(self) -> Optional[Path]:
        """สร้าง backup ก่อน deployment"""
        if not self.deployment_config["backup"]["enabled"]:
            return None

        self.logger.info("💾 Creating backup...")

        backup_dir = self.repo_path / self.deployment_config["backup"]["backup_dir"]
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"nicegold_backup_{timestamp}"
        backup_path = backup_dir / f"{backup_name}.tar.gz"

        # สร้าง tar archive
        success, output = self.run_command(
            [
                "tar",
                " - czf",
                str(backup_path),
                " -  - exclude = backups",
                " -  - exclude = logs",
                " -  - exclude = __pycache__",
                " -  - exclude = *.pyc",
                ".",
            ]
        )

        if success:
            self.logger.info(f"✅ Backup created: {backup_path}")

            # ลบ backup เก่า
            self.cleanup_old_backups(backup_dir)

            return backup_path
        else:
            self.logger.error(f"❌ Backup failed: {output}")
            return None

    def cleanup_old_backups(self, backup_dir: Path):
        """ลบ backup เก่า"""
        keep_backups = self.deployment_config["backup"]["keep_backups"]

        backup_files = sorted(
            backup_dir.glob("nicegold_backup_*.tar.gz"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if len(backup_files) > keep_backups:
            for old_backup in backup_files[keep_backups:]:
                try:
                    old_backup.unlink()
                    self.logger.info(f"🗑️ Deleted old backup: {old_backup.name}")
                except Exception as e:
                    self.logger.warning(f"Could not delete {old_backup}: {e}")

    def git_commit_and_push(self, message: str = None) -> bool:
        """Commit และ Push ไปยัง Git repository"""
        if not self.deployment_config["git"]["auto_push"]:
            self.logger.info("🚫 Auto push disabled")
            return True

        self.logger.info("📝 Committing and pushing to Git...")

        # ตั้งค่า Git user ถ้ายังไม่ได้ตั้ง
        success, _ = self.run_command(["git", "config", "user.name"])
        if not success:
            self.run_command(["git", "config", "user.name", "NICEGOLD Administrator"])

        success, _ = self.run_command(["git", "config", "user.email"])
        if not success:
            self.run_command(["git", "config", "user.email", "admin@nicegold.local"])

        # Add files
        success, _ = self.run_command(["git", "add", "."])
        if not success:
            self.logger.error("❌ Git add failed")
            return False

        # Create commit message
        if not message:
            timestamp = datetime.now().strftime("%Y - %m - %d %H:%M:%S")
            prefix = self.deployment_config["git"]["commit_prefix"]
            message = f"{prefix} Enterprise Auto - Deploy - {timestamp}"

        # Commit
        success, output = self.run_command(["git", "commit", " - m", message])
        if not success and "nothing to commit" not in output:
            self.logger.error(f"❌ Git commit failed: {output}")
            return False

        # Push
        branch = self.deployment_config["git"]["branch"]
        success, output = self.run_command(["git", "push", "origin", branch])

        if success:
            self.logger.info(f"✅ Successfully pushed to {branch}")
            return True
        else:
            self.logger.error(f"❌ Git push failed: {output}")

            # ลอง force push
            self.logger.info("🔄 Trying force push...")
            success, output = self.run_command(
                ["git", "push", " -  - force - with - lease", "origin", branch]
            )

            if success:
                self.logger.info(f"✅ Force push successful")
                return True
            else:
                self.logger.error(f"❌ Force push failed: {output}")
                return False

    def send_notification(self, success: bool, message: str):
        """ส่งการแจ้งเตือน"""
        if not self.deployment_config["notifications"]["enabled"]:
            return

        if success and not self.deployment_config["notifications"]["on_success"]:
            return

        if not success and not self.deployment_config["notifications"]["on_failure"]:
            return

        status = "✅ SUCCESS" if success else "❌ FAILED"
        timestamp = datetime.now().strftime("%Y - %m - %d %H:%M:%S")

        notification = f"""
🚀 NICEGOLD Deployment {status}
⏰ Time: {timestamp}
📝 Message: {message}
"""

        # บันทึกลง log file
        notification_file = self.repo_path / "logs" / "deployment" / "notifications.log"
        notification_file.parent.mkdir(parents=True, exist_ok=True)

        with open(notification_file, "a", encoding="utf - 8") as f:
            f.write(f"{timestamp} - {notification}\n")

        self.logger.info(f"📢 Notification: {message}")

    def generate_deployment_report(self, start_time: datetime, success: bool) -> Dict:
        """สร้างรายงาน deployment"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = {
            "deployment_id": f"deploy_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "success": success,
            "config": self.deployment_config,
            "repository_path": str(self.repo_path),
            "git_info": self.get_git_info(),
        }

        return report

    def get_git_info(self) -> Dict:
        """ดึงข้อมูล Git"""
        git_info = {}

        # Current branch
        success, branch = self.run_command(["git", "branch", " -  - show - current"])
        if success:
            git_info["branch"] = branch

        # Last commit
        success, commit = self.run_command(["git", "log", " - 1", " -  - oneline"])
        if success:
            git_info["last_commit"] = commit

        # Remote URL
        success, remote = self.run_command(["git", "remote", "get - url", "origin"])
        if success:
            git_info["remote_url"] = remote

        return git_info

    def save_deployment_report(self, report: Dict):
        """บันทึกรายงาน deployment"""
        reports_dir = self.repo_path / "reports" / "deployment"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_file = reports_dir / f"{report['deployment_id']}.json"

        with open(report_file, "w", encoding="utf - 8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📊 Deployment report saved: {report_file}")

    def deploy(self, commit_message: str = None) -> bool:
        """เริ่มการ deployment"""
        start_time = datetime.now()

        self.logger.info("🚀 Starting NICEGOLD deployment...")

        try:
            # 1. Create backup
            backup_path = self.create_backup()

            # 2. Cleanup repository
            self.cleanup_repository()

            # 3. Validate deployment
            if not self.validate_deployment():
                self.send_notification(False, "Validation failed")
                return False

            # 4. Git commit and push
            if not self.git_commit_and_push(commit_message):
                self.send_notification(False, "Git push failed")
                return False

            # 5. Save configuration
            self.save_deployment_config()

            # Success
            self.logger.info("🎉 Deployment completed successfully!")

            # Generate and save report
            report = self.generate_deployment_report(start_time, True)
            self.save_deployment_report(report)

            self.send_notification(True, "Deployment completed successfully")

            return True

        except Exception as e:
            self.logger.error(f"💥 Deployment failed: {e}")

            # Generate failure report
            report = self.generate_deployment_report(start_time, False)
            report["error"] = str(e)
            self.save_deployment_report(report)

            self.send_notification(False, f"Deployment failed: {e}")

            return False


def main():
    """ฟังก์ชันหลักสำหรับ CLI"""

    parser = argparse.ArgumentParser(description="NICEGOLD Auto Deployment System")
    parser.add_argument(" -  - message", " - m", help="Commit message")
    parser.add_argument(" -  - config", help="Path to deployment config file")
    parser.add_argument(
        " -  - no - backup", action="store_true", help="Skip backup creation"
    )
    parser.add_argument(" -  - no - push", action="store_true", help="Skip Git push")
    parser.add_argument(
        " -  - no - validation", action="store_true", help="Skip validation"
    )
    parser.add_argument(
        " -  - dry - run", action="store_true", help="Dry run (show what would be done)"
    )

    args = parser.parse_args()

    # สร้าง deployment instance
    deployment = NiceGoldDeployment()

    # ปรับแต่ง config ตาม arguments
    if args.no_backup:
        deployment.deployment_config["backup"]["enabled"] = False
    if args.no_push:
        deployment.deployment_config["git"]["auto_push"] = False
    if args.no_validation:
        deployment.deployment_config["validation"] = {
            "run_tests": False,
            "check_syntax": False,
            "check_dependencies": False,
        }

    if args.dry_run:
        logging.info("🔍 Dry run mode - showing configuration:")
        logging.info(
            json.dumps(deployment.deployment_config, indent=2, ensure_ascii=False)
        )
        return

    # เริ่ม deployment
    success = deployment.deploy(args.message)

    if success:
        logging.info("🎉 Deployment completed successfully!")
        sys.exit(0)
    else:
        logging.error("❌ Deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
