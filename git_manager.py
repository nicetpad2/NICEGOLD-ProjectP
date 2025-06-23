#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NICEGOLD Git Repository Manager
ระบบจัดการ Git repository สำหรับ NICEGOLD Enterprise
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class GitManager:
    """ระบบจัดการ Git repository สำหรับ NICEGOLD"""
    
    def __init__(self, repo_path: str = None):
        self.repo_path = repo_path or os.getcwd()
        self.setup_logging()
        
    def setup_logging(self):
        """ตั้งค่า logging"""
        log_dir = Path(self.repo_path) / "logs" / "git"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"git_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_git_command(self, command: List[str], capture_output: bool = True) -> Tuple[bool, str]:
        """เรียกใช้คำสั่ง Git"""
        try:
            full_command = ["git"] + command
            result = subprocess.run(
                full_command,
                cwd=self.repo_path,
                capture_output=capture_output,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Git command succeeded: {' '.join(command)}")
                return True, result.stdout.strip()
            else:
                self.logger.error(f"❌ Git command failed: {' '.join(command)}")
                self.logger.error(f"Error: {result.stderr}")
                return False, result.stderr.strip()
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"⏰ Git command timed out: {' '.join(command)}")
            return False, "Command timed out"
        except Exception as e:
            self.logger.error(f"💥 Git command error: {e}")
            return False, str(e)

    def check_git_status(self) -> Dict:
        """ตรวจสอบสถานะ Git repository"""
        status = {
            "is_git_repo": False,
            "current_branch": None,
            "remote_url": None,
            "status": None,
            "staged_files": [],
            "modified_files": [],
            "untracked_files": [],
            "ahead_behind": {"ahead": 0, "behind": 0}
        }
        
        # ตรวจสอบว่าเป็น Git repo หรือไม่
        success, _ = self.run_git_command(["status", "--porcelain"])
        if not success:
            return status
            
        status["is_git_repo"] = True
        
        # ตรวจสอบ branch ปัจจุบัน
        success, branch = self.run_git_command(["branch", "--show-current"])
        if success:
            status["current_branch"] = branch
            
        # ตรวจสอบ remote URL
        success, url = self.run_git_command(["remote", "get-url", "origin"])
        if success:
            status["remote_url"] = url
            
        # ตรวจสอบสถานะไฟล์
        success, git_status = self.run_git_command(["status", "--porcelain"])
        if success:
            status["status"] = git_status
            self._parse_git_status(git_status, status)
            
        # ตรวจสอบ ahead/behind
        success, ahead_behind = self.run_git_command([
            "rev-list", "--left-right", "--count", "HEAD...@{upstream}"
        ])
        if success and ahead_behind:
            try:
                ahead, behind = ahead_behind.split('\t')
                status["ahead_behind"] = {"ahead": int(ahead), "behind": int(behind)}
            except:
                pass
        
        return status

    def _parse_git_status(self, git_status: str, status: Dict):
        """แปลงผล git status"""
        for line in git_status.split('\n'):
            if not line.strip():
                continue
                
            status_code = line[:2]
            filename = line[3:]
            
            if status_code[0] in ['A', 'M', 'D', 'R', 'C']:
                status["staged_files"].append(filename)
            elif status_code[1] in ['M', 'D']:
                status["modified_files"].append(filename)
            elif status_code == '??':
                status["untracked_files"].append(filename)

    def setup_git_user(self, name: str = None, email: str = None):
        """ตั้งค่าผู้ใช้ Git"""
        if not name:
            name = "NICEGOLD Administrator"
        if not email:
            email = "admin@nicegold.local"
            
        self.run_git_command(["config", "user.name", name])
        self.run_git_command(["config", "user.email", email])
        
        self.logger.info(f"✅ Git user configured: {name} <{email}>")

    def add_files(self, files: List[str] = None) -> bool:
        """เพิ่มไฟล์เข้า staging area"""
        if files is None:
            # Add all files
            success, _ = self.run_git_command(["add", "."])
        else:
            # Add specific files
            success, _ = self.run_git_command(["add"] + files)
            
        if success:
            self.logger.info(f"✅ Files added to staging: {files or 'all files'}")
        return success

    def commit_changes(self, message: str, description: str = None) -> bool:
        """Commit การเปลี่ยนแปลง"""
        full_message = message
        if description:
            full_message = f"{message}\n\n{description}"
            
        success, output = self.run_git_command(["commit", "-m", full_message])
        
        if success:
            self.logger.info(f"✅ Changes committed: {message}")
        return success

    def push_to_remote(self, branch: str = None, force: bool = False) -> bool:
        """Push changes ไปยัง remote repository"""
        if not branch:
            branch = "main"
            
        cmd = ["push", "origin", branch]
        if force:
            cmd.insert(1, "--force-with-lease")
            
        success, output = self.run_git_command(cmd)
        
        if success:
            self.logger.info(f"✅ Successfully pushed to {branch}")
        else:
            self.logger.error(f"❌ Failed to push to {branch}: {output}")
            
        return success

    def pull_from_remote(self, branch: str = None) -> bool:
        """Pull changes จาก remote repository"""
        if not branch:
            branch = "main"
            
        success, output = self.run_git_command(["pull", "origin", branch])
        
        if success:
            self.logger.info(f"✅ Successfully pulled from {branch}")
        else:
            self.logger.error(f"❌ Failed to pull from {branch}: {output}")
            
        return success

    def create_and_push_branch(self, branch_name: str) -> bool:
        """สร้าง branch ใหม่และ push"""
        # สร้าง branch ใหม่
        success, _ = self.run_git_command(["checkout", "-b", branch_name])
        if not success:
            return False
            
        # Push branch ใหม่
        success, _ = self.run_git_command(["push", "-u", "origin", branch_name])
        
        if success:
            self.logger.info(f"✅ Created and pushed new branch: {branch_name}")
        return success

    def sync_with_remote(self) -> bool:
        """ซิงค์กับ remote repository"""
        self.logger.info("🔄 Syncing with remote repository...")
        
        # Fetch ข้อมูลล่าสุด
        success, _ = self.run_git_command(["fetch", "origin"])
        if not success:
            return False
            
        # Pull การเปลี่ยนแปลงล่าสุด
        success = self.pull_from_remote()
        
        return success

    def smart_commit_and_push(self, 
                             message: str = None, 
                             description: str = None,
                             files: List[str] = None) -> bool:
        """Commit และ Push อย่างอัจฉริยะ"""
        
        # ตรวจสอบสถานะ
        status = self.check_git_status()
        
        if not status["is_git_repo"]:
            self.logger.error("❌ Not a git repository")
            return False
            
        # ตั้งค่า default commit message
        if not message:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"🚀 NICEGOLD Update - {timestamp}"
            
        if not description:
            total_files = (len(status["staged_files"]) + 
                          len(status["modified_files"]) + 
                          len(status["untracked_files"]))
            description = f"Updated {total_files} files in NICEGOLD Enterprise system"
        
        try:
            # 1. Add files
            if not self.add_files(files):
                return False
                
            # 2. Commit changes
            if not self.commit_changes(message, description):
                return False
                
            # 3. Sync with remote (pull latest changes)
            if not self.sync_with_remote():
                self.logger.warning("⚠️ Failed to sync with remote, continuing with push...")
                
            # 4. Push to remote
            if not self.push_to_remote():
                return False
                
            self.logger.info("🎉 Successfully committed and pushed changes!")
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Error during commit and push: {e}")
            return False

    def generate_status_report(self) -> Dict:
        """สร้างรายงานสถานะ Git"""
        status = self.check_git_status()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "repository_status": status,
            "summary": {
                "total_staged": len(status.get("staged_files", [])),
                "total_modified": len(status.get("modified_files", [])),
                "total_untracked": len(status.get("untracked_files", [])),
                "needs_push": status.get("ahead_behind", {}).get("ahead", 0) > 0,
                "needs_pull": status.get("ahead_behind", {}).get("behind", 0) > 0
            }
        }
        
        return report

    def save_status_report(self, filename: str = None):
        """บันทึกรายงานสถานะ"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"git_status_report_{timestamp}.json"
            
        report = self.generate_status_report()
        
        reports_dir = Path(self.repo_path) / "reports" / "git"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"📊 Status report saved: {report_path}")
        return report_path


def main():
    """ฟังก์ชันหลักสำหรับใช้งานแบบ CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NICEGOLD Git Manager")
    parser.add_argument("--action", 
                       choices=["status", "commit", "push", "sync", "smart-push"], 
                       default="status",
                       help="Action to perform")
    parser.add_argument("--message", "-m", help="Commit message")
    parser.add_argument("--description", "-d", help="Commit description")
    parser.add_argument("--files", nargs="*", help="Specific files to add")
    parser.add_argument("--force", action="store_true", help="Force push")
    parser.add_argument("--report", action="store_true", help="Generate status report")
    
    args = parser.parse_args()
    
    # สร้าง GitManager instance
    git_manager = GitManager()
    
    if args.action == "status":
        status = git_manager.check_git_status()
        print("\n🔍 Git Repository Status:")
        print(f"├── Repository: {'✅' if status['is_git_repo'] else '❌'}")
        print(f"├── Branch: {status.get('current_branch', 'Unknown')}")
        print(f"├── Remote: {status.get('remote_url', 'Not configured')}")
        print(f"├── Staged files: {len(status.get('staged_files', []))}")
        print(f"├── Modified files: {len(status.get('modified_files', []))}")
        print(f"├── Untracked files: {len(status.get('untracked_files', []))}")
        print(f"└── Ahead/Behind: +{status.get('ahead_behind', {}).get('ahead', 0)}/-{status.get('ahead_behind', {}).get('behind', 0)}")
        
    elif args.action == "commit":
        if not args.message:
            print("❌ Commit message required for commit action")
            return
        git_manager.add_files(args.files)
        git_manager.commit_changes(args.message, args.description)
        
    elif args.action == "push":
        git_manager.push_to_remote(force=args.force)
        
    elif args.action == "sync":
        git_manager.sync_with_remote()
        
    elif args.action == "smart-push":
        git_manager.smart_commit_and_push(
            message=args.message,
            description=args.description,
            files=args.files
        )
    
    if args.report:
        git_manager.save_status_report()


if __name__ == "__main__":
    main()
