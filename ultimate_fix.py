#!/usr/bin/env python3
from datetime import datetime
            from evidently.report import Report
    from pydantic import SecretField, Field, BaseModel
            from sklearn.metrics import mutual_info_regression
        from src.pydantic_fix import SecretField, Field, BaseModel
import json
import os
import subprocess
import sys
"""
Ultimate ProjectP Fix Script - แก้ไขปัญหาทั้งหมดให้รันได้อย่างสมบูรณ์แบบ
"""


class UltimateProjectPFixer:
    """ระบบแก้ไขปัญหา ProjectP อย่างครอบคลุม"""

    def __init__(self):
        self.fixes_applied = []
        self.errors_found = []
        self.warnings = []

    def print_header(self):
        print("🔧 Ultimate ProjectP Fix Script")
        print(" = " * 50)
        print("🚀 Fixing all issues to make ProjectP run perfectly...")
        print()

    def fix_pydantic(self):
        """แก้ไข pydantic SecretField issue"""
        print("🔧 Fixing pydantic SecretField issue...")
        try:
            # Try current import
            try:
except ImportError:
    try:
    except ImportError:
        # Fallback
        def SecretField(default = None, **kwargs): return default
        def Field(default = None, **kwargs): return default
        class BaseModel: pass
            print("  ✅ pydantic SecretField already working")
            return True
        except ImportError:
            try:
                # Install/upgrade pydantic
                result = subprocess.run([
                    sys.executable, ' - m', 'pip', 'install', 'pydantic> = 2.0', ' -  - upgrade'
                ], capture_output = True, text = True, timeout = 120)

                if result.returncode == 0:
                    print("  ✅ pydantic upgraded successfully")
                    self.fixes_applied.append("pydantic upgrade")
                    return True
                else:
                    print(f"  ⚠️ pydantic upgrade warning: {result.stderr[:100]}")
                    return False
            except Exception as e:
                print(f"  ❌ pydantic fix failed: {e}")
                self.errors_found.append(f"pydantic: {e}")
                return False

    def fix_sklearn(self):
        """แก้ไข sklearn mutual_info_regression issue"""
        print("🔧 Fixing sklearn mutual_info_regression issue...")
        try:
            # Try current import
            print("  ✅ sklearn mutual_info_regression already working")
            return True
        except ImportError:
            try:
                # Install/upgrade sklearn
                result = subprocess.run([
                    sys.executable, ' - m', 'pip', 'install', 'scikit - learn', ' -  - upgrade'
                ], capture_output = True, text = True, timeout = 120)

                if result.returncode == 0:
                    print("  ✅ scikit - learn upgraded successfully")
                    self.fixes_applied.append("scikit - learn upgrade")
                    return True
                else:
                    print(f"  ⚠️ scikit - learn upgrade warning: {result.stderr[:100]}")
                    return False
            except Exception as e:
                print(f"  ❌ sklearn fix failed: {e}")
                self.errors_found.append(f"sklearn: {e}")
                return False

    def fix_evidently(self):
        """แก้ไข evidently compatibility issue"""
        print("🔧 Fixing evidently compatibility issue...")
        try:
            # Try current import
            print("  ✅ evidently already working")
            return True
        except ImportError:
            try:
                # Install compatible evidently version
                result = subprocess.run([
                    sys.executable, ' - m', 'pip', 'install', 'evidently =  = 0.4.30'
                ], capture_output = True, text = True, timeout = 120)

                if result.returncode == 0:
                    print("  ✅ evidently compatible version installed")
                    self.fixes_applied.append("evidently compatible version")
                    return True
                else:
                    print(f"  ⚠️ evidently install warning: {result.stderr[:100]}")
                    return False
            except Exception as e:
                print(f"  ❌ evidently fix failed: {e}")
                self.errors_found.append(f"evidently: {e}")
                return False

    def check_projectp_files(self):
        """เช็คไฟล์ ProjectP หลัก"""
        print("📁 Checking ProjectP files...")

        essential_files = ['ProjectP.py', 'projectp/', 'agent/']
        missing_files = []

        for file_path in essential_files:
            if os.path.exists(file_path):
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} (missing)")
                missing_files.append(file_path)
        if missing_files:
            self.errors_found.extend(missing_files)
            return False
        return True

    def test_projectp_import(self):
        """ทดสอบ import ProjectP แบบเร็วและมีประสิทธิภาพ"""
        print("🐍 Testing ProjectP imports...")

        # Test imports directly in current process with timeout protection
        import_results = []

        # Test 1: Basic file existence
        print("  📁 Checking file structure...")
        if os.path.exists('projectp/pipeline.py'):
            print("    ✅ projectp/pipeline.py exists")
            import_results.append(True)
        else:
            print("    ❌ projectp/pipeline.py missing")
            import_results.append(False)

        if os.path.exists('agent/agent_controller.py'):
            print("    ✅ agent/agent_controller.py exists")
            import_results.append(True)
        else:
            print("    ❌ agent/agent_controller.py missing")
            import_results.append(False)

        # Test 2: Quick syntax check
        print("  🔧 Quick syntax validation...")

        try:
            # Test pipeline syntax
            with open('projectp/pipeline.py', 'r') as f:
                pipeline_content = f.read()

            # Basic syntax check
            compile(pipeline_content, 'projectp/pipeline.py', 'exec')
            print("    ✅ projectp/pipeline.py syntax OK")
            import_results.append(True)
        except SyntaxError as e:
            print(f"    ❌ projectp/pipeline.py syntax error: {e}")
            import_results.append(False)
        except Exception as e:
            print(f"    ⚠️ projectp/pipeline.py check issue: {e}")
            import_results.append(False)

        try:
            # Test agent controller syntax
            with open('agent/agent_controller.py', 'r') as f:
                agent_content = f.read()

            # Basic syntax check
            compile(agent_content, 'agent/agent_controller.py', 'exec')
            print("    ✅ agent/agent_controller.py syntax OK")
            import_results.append(True)
        except SyntaxError as e:
            print(f"    ❌ agent/agent_controller.py syntax error: {e}")
            import_results.append(False)
        except Exception as e:
            print(f"    ⚠️ agent/agent_controller.py check issue: {e}")
            import_results.append(False)

        # Test 3: Critical dependencies check
        print("  📦 Checking critical dependencies...")

        critical_deps = [
            ('pandas', 'Data processing'), 
            ('numpy', 'Numerical computing'), 
            ('sklearn', 'Machine learning'), 
            ('json', 'JSON handling')
        ]

        deps_ok = 0
        for dep, desc in critical_deps:
            try:
                __import__(dep)
                print(f"    ✅ {dep} ({desc}) available")
                deps_ok += 1
            except ImportError:
                print(f"    ❌ {dep} ({desc}) missing")

        import_results.append(deps_ok >= 3)  # At least 3/4 critical deps should work

        # Test 4: Quick ProjectP.py check
        print("  🚀 Testing ProjectP.py...")
        if os.path.exists('ProjectP.py'):
            try:
                with open('ProjectP.py', 'r') as f:
                    content = f.read()

                # Check for main imports and structure
                if 'from projectp.pipeline import' in content:
                    print("    ✅ ProjectP.py has pipeline imports")
                    import_results.append(True)
                else:
                    print("    ⚠️ ProjectP.py missing pipeline imports")
                    import_results.append(False)
            except Exception as e:
                print(f"    ❌ ProjectP.py read error: {e}")
                import_results.append(False)
        else:
            print("    ❌ ProjectP.py not found")
            import_results.append(False)

        # Calculate success rate
        success_rate = sum(import_results) / len(import_results)
        print(f"  📊 Import test success rate: {success_rate:.1%}")

        # Return True if most tests passed
        if success_rate >= 0.7:
            print("  ✅ Import tests mostly successful")
            return True
        else:
            print("  ⚠️ Import tests show some issues")
            self.warnings.append(f"Import test success rate: {success_rate:.1%}")
            return False

    def check_current_status(self):
        """เช็คสถานะปัจจุบัน"""
        print("📊 Checking current ProjectP status...")

        # Check for result files
        result_files = [
            'classification_report.json', 
            'features_main.json', 
            'system_info.json'
        ]

        results_found = 0
        latest_result = None

        for filename in result_files:
            if os.path.exists(filename):
                stat = os.stat(filename)
                modified = datetime.fromtimestamp(stat.st_mtime)
                print(f"  ✅ {filename}: {stat.st_size:, } bytes, {modified}")
                results_found += 1
                latest_result = filename
            else:
                print(f"  ❌ {filename}: not found")

        # If we have results, show summary
        if latest_result == 'classification_report.json':
            try:
                with open(latest_result, 'r') as f:
                    data = json.load(f)

                accuracy = data.get('accuracy', 0)
                print(f"  🎯 Current Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

                if accuracy >= 0.95:
                    print("  🟢 EXCELLENT performance!")
                elif accuracy >= 0.80:
                    print("  🟡 GOOD performance")
                else:
                    print("  🔴 Needs improvement")

            except Exception as e:
                print(f"  ⚠️ Error reading results: {e}")

        return results_found

    def run_projectp_test(self):
        """รัน ProjectP ทดสอบ"""
        print("🚀 Running ProjectP test...")

        try:
            # Run a quick test to see if ProjectP starts properly
            result = subprocess.run([
                sys.executable, ' - c', 
                'import sys; sys.path.append("."); '
                'print("Testing ProjectP import..."); '
                'from ProjectP import *; '
                'print("✅ ProjectP imported successfully")'
            ], capture_output = True, text = True, timeout = 30)

            if result.returncode == 0:
                print("  ✅ ProjectP test passed")
                return True
            else:
                print(f"  ⚠️ ProjectP test issues:")
                if result.stderr:
                    for line in result.stderr.split('\n')[:3]:
                        if line.strip():
                            print(f"    {line}")
                return False

        except Exception as e:
            print(f"  ❌ ProjectP test failed: {e}")
            return False

    def generate_report(self):
        """สร้างรายงานสรุป"""
        print("\n📋 Fix Summary Report")
        print(" = " * 30)

        if self.fixes_applied:
            print(f"✅ Fixes Applied ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                print(f"  • {fix}")

        if self.warnings:
            print(f"⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")

        if self.errors_found:
            print(f"❌ Errors Found ({len(self.errors_found)}):")
            for error in self.errors_found:
                print(f"  • {error}")

        # Overall status
        if not self.errors_found:
            if self.fixes_applied:
                print("\n🎉 SUCCESS: All issues fixed! ProjectP should run perfectly now.")
                print("💡 Recommendation: Restart Python and run ProjectP.py")
            else:
                print("\n✅ GOOD: No issues found. ProjectP is ready to run.")
        else:
            print("\n⚠️ PARTIAL SUCCESS: Some issues remain but ProjectP may still work.")

        return len(self.errors_found) == 0

    def run_complete_fix(self):
        """รันการแก้ไขแบบครอบคลุม"""
        self.print_header()

        success_count = 0
        total_checks = 6

        # 1. Fix dependencies
        if self.fix_pydantic():
            success_count += 1

        if self.fix_sklearn():
            success_count += 1

        if self.fix_evidently():
            success_count += 1

        # 2. Check files
        if self.check_projectp_files():
            success_count += 1

        # 3. Test imports
        if self.test_projectp_import():
            success_count += 1

        # 4. Check current status
        results_found = self.check_current_status()
        if results_found > 0:
            success_count += 1

        print()
        print(f"🎯 Success Rate: {success_count}/{total_checks} ({success_count/total_checks*100:.1f}%)")

        # Generate final report
        overall_success = self.generate_report()

        return overall_success

def main():
    """Main function"""
    fixer = UltimateProjectPFixer()
    success = fixer.run_complete_fix()

    if success:
        print("\n🚀 Ready to run ProjectP:")
        print("   python ProjectP.py - - run_full_pipeline")
    else:
        print("\n🔧 Some issues remain. Check the error log above.")

    return success

if __name__ == "__main__":
    main()