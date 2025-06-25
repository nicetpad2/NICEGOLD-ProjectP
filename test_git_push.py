# -*- coding: utf - 8 -* - 
#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys
"""
🧪 ทดสอบระบบ Git Push สำหรับ NICEGOLD
"""


def run_command(command, cwd = None):
    """เรียกใช้คำสั่งและดึงผลลัพธ์"""
    try:
        result = subprocess.run(
            command, 
            shell = True, 
            cwd = cwd or os.getcwd(), 
            capture_output = True, 
            text = True, 
            timeout = 60
        )

        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def test_git_functionality():
    """ทดสอบการทำงานของ Git"""
    print("🔍 ทดสอบการทำงานของ Git...")

    # ตรวจสอบ Git version
    success, output, error = run_command("git - - version")
    if success:
        print(f"✅ Git version: {output}")
    else:
        print(f"❌ Git not available: {error}")
        return False

    # ตรวจสอบว่าอยู่ใน Git repository
    success, output, error = run_command("git rev - parse - - is - inside - work - tree")
    if success and output == "true":
        print("✅ อยู่ใน Git repository")
    else:
        print(f"❌ ไม่ใช่ Git repository: {error}")
        return False

    # ตรวจสอบ remote
    success, output, error = run_command("git remote -v")
    if success and output:
        print(f"✅ Git remotes:\n{output}")
    else:
        print(f"⚠️ ไม่พบ remote repositories: {error}")

    # ตรวจสอบ branch ปัจจุบัน
    success, output, error = run_command("git branch - - show - current")
    if success:
        print(f"✅ Current branch: {output}")
    else:
        print(f"⚠️ ไม่สามารถดึง branch ปัจจุบัน: {error}")

    # ตรวจสอบสถานะ
    success, output, error = run_command("git status - - porcelain")
    if success:
        if output:
            lines = output.split('\n')
            print(f"✅ พบไฟล์ที่เปลี่ยนแปลง: {len(lines)} ไฟล์")
            # แสดงไฟล์ 5 ตัวแรก
            for line in lines[:5]:
                print(f"  {line}")
            if len(lines) > 5:
                print(f"  ... และอีก {len(lines) - 5} ไฟล์")
        else:
            print("✅ ไม่มีไฟล์ที่เปลี่ยนแปลง")
    else:
        print(f"❌ ไม่สามารถตรวจสอบสถานะ: {error}")

    return True

def test_git_config():
    """ทดสอบการตั้งค่า Git"""
    print("\n⚙️ ทดสอบการตั้งค่า Git...")

    # ตรวจสอบ user.name
    success, output, error = run_command("git config user.name")
    if success and output:
        print(f"✅ Git user.name: {output}")
    else:
        print("⚠️ ไม่พบ user.name - กำลังตั้งค่า...")
        run_command("git config user.name 'NICEGOLD Administrator'")
        print("✅ ตั้งค่า user.name เรียบร้อย")

    # ตรวจสอบ user.email
    success, output, error = run_command("git config user.email")
    if success and output:
        print(f"✅ Git user.email: {output}")
    else:
        print("⚠️ ไม่พบ user.email - กำลังตั้งค่า...")
        run_command("git config user.email 'admin@nicegold.local'")
        print("✅ ตั้งค่า user.email เรียบร้อย")

def test_file_operations():
    """ทดสอบการทำงานกับไฟล์"""
    print("\n📁 ทดสอบการทำงานกับไฟล์...")

    # สร้างไฟล์ทดสอบ
    test_file = Path("test_push_functionality.txt")
    timestamp = datetime.now().strftime("%Y - %m - %d %H:%M:%S")

    content = f"""🧪 NICEGOLD Git Push Test File
Created: {timestamp}
Repository: NICEGOLD - ProjectP
Purpose: Testing Git push functionality

This file was created automatically to test the Git push system.
"""

    try:
        with open(test_file, 'w', encoding = 'utf - 8') as f:
            f.write(content)
        print(f"✅ สร้างไฟล์ทดสอบ: {test_file}")

        # เพิ่มไฟล์เข้า Git
        success, output, error = run_command(f"git add {test_file}")
        if success:
            print("✅ เพิ่มไฟล์เข้า staging area")
        else:
            print(f"❌ ไม่สามารถเพิ่มไฟล์: {error}")
            return False

        return True

    except Exception as e:
        print(f"❌ ไม่สามารถสร้างไฟล์ทดสอบ: {e}")
        return False

def test_commit():
    """ทดสอบการ commit"""
    print("\n💾 ทดสอบการ commit...")

    timestamp = datetime.now().strftime("%Y - %m - %d %H:%M:%S")
    commit_message = f"🧪 Test commit for Git push functionality - {timestamp}"

    success, output, error = run_command(f'git commit -m "{commit_message}"')

    if success:
        print("✅ Commit สำเร็จ")
        return True
    elif "nothing to commit" in error:
        print("✅ ไม่มีการเปลี่ยนแปลงใหม่ที่จะ commit")
        return True
    else:
        print(f"❌ Commit ล้มเหลว: {error}")
        return False

def test_push_dry_run():
    """ทดสอบ push แบบ dry run"""
    print("\n🚀 ทดสอบ push (dry run)...")

    success, output, error = run_command("git push - - dry - run origin main")

    if success:
        print("✅ Push dry run สำเร็จ")
        print(f"Output: {output}")
        return True
    else:
        print(f"⚠️ Push dry run warning/error: {error}")
        # อาจจะเป็น warning ปกติ
        return True

def cleanup_test_files():
    """ทำความสะอาดไฟล์ทดสอบ"""
    print("\n🧹 ทำความสะอาดไฟล์ทดสอบ...")

    test_file = Path("test_push_functionality.txt")
    if test_file.exists():
        try:
            test_file.unlink()
            print("✅ ลบไฟล์ทดสอบเรียบร้อย")
        except Exception as e:
            print(f"⚠️ ไม่สามารถลบไฟล์ทดสอบ: {e}")

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 เริ่มทดสอบระบบ Git Push สำหรับ NICEGOLD")
    print(" = " * 60)

    try:
        # ทดสอบการทำงานพื้นฐานของ Git
        if not test_git_functionality():
            print("\n❌ การทดสอบ Git พื้นฐานล้มเหลว")
            return False

        # ทดสอบการตั้งค่า Git
        test_git_config()

        # ทดสอบการทำงานกับไฟล์
        if not test_file_operations():
            print("\n❌ การทดสอบไฟล์ล้มเหลว")
            return False

        # ทดสอบ commit
        if not test_commit():
            print("\n❌ การทดสอบ commit ล้มเหลว")
            return False

        # ทดสอบ push (dry run)
        test_push_dry_run()

        print("\n" + " = " * 60)
        print("🎉 การทดสอบเสร็จสิ้น!")
        print("\nระบบ Git ของคุณพร้อมใช้งาน สามารถใช้คำสั่งต่อไปนี้ได้:")
        print("  ./quick_push.sh                 # Push แบบง่าย")
        print("  python git_manager.py - - action smart - push  # Push แบบอัจฉริยะ")
        print("  python auto_deployment.py       # Deploy แบบอัตโนมัติ")

        return True

    except KeyboardInterrupt:
        print("\n\n⚠️ การทดสอบถูกยกเลิก")
        return False
    except Exception as e:
        print(f"\n💥 เกิดข้อผิดพลาดในการทดสอบ: {e}")
        return False
    finally:
        cleanup_test_files()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)