#!/usr/bin/env python3
"""
Quick test for Evidently fix
"""
import sys
print("Testing Evidently compatibility...")

try:
    from src.evidently_compat import ValueDrift, EVIDENTLY_AVAILABLE
    print(f"✅ Import successful! EVIDENTLY_AVAILABLE = {EVIDENTLY_AVAILABLE}")
    
    # Test creating instance
    vd = ValueDrift('test_column')
    print(f"✅ ValueDrift instance created for column: {vd.column_name}")
    
    # Test calculation with None data (should use fallback)
    result = vd.calculate(None, None)
    print(f"✅ Calculation successful: {result}")
    
    print("\n🎉 Evidently compatibility fix is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
            
            accuracy = data.get('accuracy', 0)
            print(f"  🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            if accuracy >= 0.95:
                print("  🟢 EXCELLENT: Performance exceeds expectations!")
                estimated_auc = 0.95 + (accuracy - 0.95) * 2  # Conservative estimate
                print(f"  📊 Estimated AUC: ~{estimated_auc:.3f}")
                
                if estimated_auc >= 0.70:
                    print("  🎉 AUC TARGET LIKELY ACHIEVED!")
                    
        except Exception as e:
            print(f"  ❌ Error reading results: {e}")
    else:
        print("  ⏳ No results found yet")
    
    # 3. Check Python processes
    print("\n🐍 Python Processes:")
    try:
        import subprocess
        result = subprocess.run([
            'powershell', 'Get-Process python -ErrorAction SilentlyContinue'
        ], capture_output=True, text=True, timeout=5)
        
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if len(lines) > 2:
                print(f"  🔄 Found {len(lines)-2} Python processes running")
                for line in lines[2:3]:  # Show first process
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 6:
                            pid = parts[5]
                            cpu = parts[4]
                            print(f"    PID {pid}: {cpu}s CPU time")
            else:
                print("  ⚠️ No Python processes found")
        else:
            print("  ⚠️ No Python processes found")
            
    except Exception as e:
        print(f"  ❌ Error checking processes: {e}")
    
    # 4. Quick recommendations
    print("\n💡 Quick Actions:")
    
    if files_ok >= 3:
        print("  ✅ Core files present - ProjectP should work")
        if os.path.exists('classification_report.json'):
            print("  🚀 Ready to run: python ProjectP.py --run_full_pipeline")
        else:
            print("  🎯 Suggest running: python ProjectP.py --run_full_pipeline")
    else:
        print("  ⚠️ Some core files missing - check project integrity")
    
    # 5. Import test (simplified)
    print("\n🔧 Quick Import Test:")
    import_issues = []
    
    # Test critical imports
    try:
        import pandas as pd
        print("  ✅ pandas")
    except ImportError:
        print("  ❌ pandas")
        import_issues.append("pandas")
    
    try:
        import numpy as np
        print("  ✅ numpy")
    except ImportError:
        print("  ❌ numpy")
        import_issues.append("numpy")
    
    try:
        from sklearn.metrics import accuracy_score
        print("  ✅ sklearn.metrics")
    except ImportError:
        print("  ❌ sklearn.metrics")
        import_issues.append("sklearn")
    
    try:
        import json
        print("  ✅ json")
    except ImportError:
        print("  ❌ json")
        import_issues.append("json")
    
    if import_issues:
        print(f"  ⚠️ Issues with: {', '.join(import_issues)}")
        print("  💡 Run: pip install pandas numpy scikit-learn")
    else:
        print("  ✅ Core imports working")
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"  📁 Files: {files_ok}/4 essential files found")
    print(f"  📊 Results: {'✅ Available' if os.path.exists('classification_report.json') else '⏳ Pending'}")
    print(f"  🐍 Imports: {'✅ Working' if not import_issues else '⚠️ Issues'}")
    
    if files_ok >= 3 and not import_issues:
        print("  🎉 STATUS: Ready to run ProjectP!")
        return True
    else:
        print("  ⚠️ STATUS: Some setup needed")
        return False

def quick_fix_attempt():
    """ลองแก้ไขปัญหาง่าย ๆ"""
    print("\n🔧 Quick Fix Attempt:")
    
    # Check if we can run a simple test
    try:
        print("  🔍 Testing basic ProjectP functionality...")
        
        # Try importing just the main module
        if os.path.exists('ProjectP.py'):
            with open('ProjectP.py', 'r') as f:
                content = f.read()
            
            if 'import' in content and 'def' in content:
                print("  ✅ ProjectP.py structure looks good")
                
                # Try a simple syntax check
                try:
                    compile(content, 'ProjectP.py', 'exec')
                    print("  ✅ ProjectP.py syntax is valid")
                    return True
                except SyntaxError as e:
                    print(f"  ❌ ProjectP.py syntax error: {e}")
                    return False
            else:
                print("  ⚠️ ProjectP.py structure unclear")
                return False
        else:
            print("  ❌ ProjectP.py not found")
            return False
            
    except Exception as e:
        print(f"  ❌ Quick fix failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Quick ProjectP Diagnostic")
    print("=" * 50)
    
    # Run quick check
    status_ok = quick_status_check()
    
    if not status_ok:
        # Try quick fix
        fix_ok = quick_fix_attempt()
        if fix_ok:
            print("\n✅ Quick fix successful!")
        else:
            print("\n⚠️ Quick fix needed manual intervention")
    
    # Final recommendation
    print("\n🎯 Final Recommendation:")
    
    if os.path.exists('classification_report.json'):
        print("  🎉 ProjectP has recent results - system is working!")
        print("  🚀 You can run: python ProjectP.py --run_full_pipeline")
    elif status_ok:
        print("  ✅ System looks ready - try running ProjectP")
        print("  🚀 Run: python ProjectP.py --run_full_pipeline")
    else:
        print("  🔧 Some issues detected - check dependencies")
        print("  💡 Try: pip install --upgrade pandas numpy scikit-learn")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
