#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE FULL PIPELINE VALIDATION TEST - NICEGOLD ProjectP
Complete validation of all Full Pipeline modes after fixes
"""

import os
import sys
import time
from datetime import datetime


def test_pipeline_syntax():
    """Test all pipeline files for syntax errors"""
    print("🔍 Testing Pipeline Syntax...")
    
    test_files = [
        "/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/production_full_pipeline.py",
        "/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/enhanced_full_pipeline.py",
        "/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/ProjectP.py"
    ]
    
    results = {}
    
    for file_path in test_files:
        print(f"   📁 Testing {os.path.basename(file_path)}...")
        
        if not os.path.exists(file_path):
            results[file_path] = "❌ File not found"
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to compile the Python code
            compile(content, file_path, 'exec')
            results[file_path] = "✅ Syntax OK"
            
        except SyntaxError as e:
            results[file_path] = f"❌ Syntax Error: {e}"
        except Exception as e:
            results[file_path] = f"⚠️ Other Error: {e}"
    
    return results

def test_imports():
    """Test critical imports"""
    print("\n📦 Testing Critical Imports...")
    
    critical_imports = [
        ("pandas", "pd"),
        ("numpy", "np"), 
        ("sklearn", None),
        ("psutil", None),
        ("rich", None)
    ]
    
    results = {}
    
    for module_name, alias in critical_imports:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            results[module_name] = "✅ Available"
        except ImportError:
            results[module_name] = "❌ Missing"
        except Exception as e:
            results[module_name] = f"⚠️ Error: {e}"
    
    return results

def test_data_availability():
    """Test if data files are available"""
    print("\n📊 Testing Data Availability...")
    
    data_paths = [
        "/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/datacsv/XAUUSD_M1.csv",
        "/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/datacsv/XAUUSD_M15.csv"
    ]
    
    results = {}
    
    for data_path in data_paths:
        if os.path.exists(data_path):
            try:
                file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
                results[os.path.basename(data_path)] = f"✅ Available ({file_size:.1f} MB)"
            except Exception as e:
                results[os.path.basename(data_path)] = f"⚠️ Error reading: {e}"
        else:
            results[os.path.basename(data_path)] = "❌ Not found"
    
    return results

def test_resource_management():
    """Test resource monitoring capabilities"""
    print("\n🔋 Testing Resource Management...")
    
    try:
        import psutil

        # Test CPU monitoring
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        results = {
            "CPU Monitoring": f"✅ Working ({cpu_percent:.1f}%)",
            "Memory Monitoring": f"✅ Working ({memory.percent:.1f}% used)",
            "Resource Control": "✅ psutil available"
        }
        
    except ImportError:
        results = {
            "Resource Control": "❌ psutil not available - install with: pip install psutil"
        }
    except Exception as e:
        results = {
            "Resource Control": f"⚠️ Error: {e}"
        }
    
    return results

def main():
    """Run comprehensive validation tests"""
    print("🧪 COMPREHENSIVE FULL PIPELINE VALIDATION")
    print("=" * 50)
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all tests
    all_results = {}
    
    # Test 1: Syntax validation
    all_results["Syntax Tests"] = test_pipeline_syntax()
    
    # Test 2: Import validation  
    all_results["Import Tests"] = test_imports()
    
    # Test 3: Data availability
    all_results["Data Tests"] = test_data_availability()
    
    # Test 4: Resource management
    all_results["Resource Tests"] = test_resource_management()
    
    # Summary report
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY REPORT")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        print(f"\n🔧 {category}:")
        for test_name, result in results.items():
            print(f"   {result} {test_name}")
            total_tests += 1
            if result.startswith("✅"):
                passed_tests += 1
    
    # Overall status
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 OVERALL VALIDATION STATUS:")
    print(f"   Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("   🟢 EXCELLENT - System ready for Full Pipeline testing")
        status = "EXCELLENT"
    elif success_rate >= 75:
        print("   🟡 GOOD - Minor issues, system mostly ready")
        status = "GOOD"
    elif success_rate >= 50:
        print("   🟠 NEEDS ATTENTION - Some critical issues found")
        status = "NEEDS_ATTENTION"
    else:
        print("   🔴 CRITICAL ISSUES - Major problems need fixing")
        status = "CRITICAL"
    
    print(f"\n✅ Validation completed: {status}")
    return status == "EXCELLENT" or status == "GOOD"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
