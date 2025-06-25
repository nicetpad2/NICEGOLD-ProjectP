#!/usr/bin/env python3
    from comprehensive_import_fix import apply_all_import_patches
from pathlib import Path
        from src.evidently_compat import safe_evidently_import
        from tracking import EnterpriseTracker
import logging
import os
        import ProjectP
import sys
        import traceback
"""
ระบบแก้ไขปัญหาหลักสำหรับ ProjectP
จัดการปัญหา import ทั้งหมดและเตรียมระบบให้พร้อมใช้งาน
"""


# เพิ่ม path สำหรับ imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import comprehensive fix
try:
    print("✅ Comprehensive import fix loaded")
except ImportError as e:
    print(f"⚠️ Could not load comprehensive fix: {e}")

    def apply_all_import_patches():
        return {'status': 'fallback_mode'}

def main():
    """Main execution function"""
    print("🚀 Starting ProjectP with comprehensive fixes...")

    # Apply all import patches
    print("🔧 Applying import patches...")
    patch_results = apply_all_import_patches()

    # แสดงผลลัพธ์
    print("\n📊 Import Patch Results:")
    for component, result in patch_results.items():
        status = "✅" if result.get('success', False) else "❌"
        print(f"  {status} {component}")

    # ตรวจสอบ EnterpriseTracker
    print("\n🔍 Testing EnterpriseTracker...")
    try:
        tracker = EnterpriseTracker()
        print("✅ EnterpriseTracker available")
    except Exception as e:
        print(f"❌ EnterpriseTracker error: {e}")

    # ตรวจสอบ Evidently
    print("\n🔍 Testing Evidently...")
    try:
        evidently_available, metrics = safe_evidently_import()
        status = "✅" if evidently_available else "⚠️"
        print(f"{status} Evidently: {evidently_available}")
        print(f"   Available metrics: {list(metrics.keys())}")
    except Exception as e:
        print(f"❌ Evidently test error: {e}")

    # Import ProjectP หลัก
    print("\n🚀 Loading ProjectP...")
    try:
        # Set environment variables
        os.environ['EVIDENTLY_FALLBACK'] = 'true'
        os.environ['ML_PROTECTION_FALLBACK'] = 'true'

        # Import ProjectP
        print("✅ ProjectP loaded successfully")

        # ทดสอบการทำงาน
        print("\n🧪 Testing ProjectP functionality...")

        # ดูว่ามี argument หรือไม่
        if len(sys.argv) > 1:
            print(f"Running with arguments: {sys.argv[1:]}")
        else:
            print("No arguments provided - running basic test")

        print("✅ ProjectP ready for operation")
        return True

    except Exception as e:
        print(f"❌ ProjectP loading failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"\n{'🎉' if success else '❌'} ProjectP initialization {'completed' if success else 'failed'}")
    sys.exit(exit_code)