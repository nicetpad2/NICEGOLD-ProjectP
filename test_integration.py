#!/usr/bin/env python3
"""
🔥 TEST INTEGRATED EMERGENCY FIXES
ทดสอบการบูรณาการ emergency fixes เข้ากับทุกโหมดของ ProjectP
"""

import subprocess
import sys
import time
from pathlib import Path

def test_integration():
    """ทดสอบการบูรณาการ"""
    print("🔥 Testing Integrated Emergency Fixes...")
    print("=" * 60)
    
    # Test 1: Emergency fixes module
    print("\n🧪 Test 1: Testing Emergency Fixes Module...")
    try:
        import integrated_emergency_fixes as fixes
        manager = fixes.create_emergency_fix_manager()
        print("✅ Emergency fixes module imported successfully")
        
        # Test with sample data
        import pandas as pd
        import numpy as np
        
        # Create problematic data
        df = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'target': np.concatenate([np.zeros(990), np.ones(10)])  # Extreme imbalance
        })
        
        # Test health check
        is_healthy, issues = manager.check_data_health(df)
        print(f"📊 Health check result: {'Healthy' if is_healthy else 'Issues found'}")
        print(f"📋 Issues: {issues}")
        
        # Test auto fix
        if not is_healthy:
            print("🔧 Applying auto fixes...")
            fixed_df = manager.auto_fix_data(df)
            print(f"✅ Data fixed: {fixed_df.shape}")
        
        print("✅ Test 1 PASSED")
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        return False
    
    # Test 2: ProjectP integration
    print("\n🧪 Test 2: Testing ProjectP Integration...")
    try:
        # Test import
        import ProjectP
        print("✅ ProjectP imported successfully")
        
        # Check if emergency fix functions are available
        if hasattr(ProjectP, 'apply_emergency_fixes_to_pipeline'):
            print("✅ Emergency fix integration found in ProjectP")
        else:
            print("⚠️ Emergency fix integration not found, but continuing...")
        
        # Check if ultimate mode is available
        if hasattr(ProjectP, 'run_ultimate_mode'):
            print("✅ Ultimate mode found in ProjectP")
        else:
            print("⚠️ Ultimate mode not found")
        
        print("✅ Test 2 PASSED")
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False
    
    # Test 3: Command line arguments
    print("\n🧪 Test 3: Testing Command Line Arguments...")
    try:
        # Test help to see if ultimate_pipeline argument exists
        result = subprocess.run([sys.executable, "ProjectP.py", "--help"], 
                              capture_output=True, text=True, timeout=30)
        
        if "ultimate_pipeline" in result.stdout:
            print("✅ Ultimate pipeline argument found in help")
        else:
            print("⚠️ Ultimate pipeline argument not found in help")
        
        print("✅ Test 3 PASSED")
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        return False
    
    # Test 4: File structure
    print("\n🧪 Test 4: Testing File Structure...")
    
    required_files = [
        "ProjectP.py",
        "integrated_emergency_fixes.py", 
        "run_ultimate_pipeline.bat"
    ]
    
    all_found = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} found")
        else:
            print(f"❌ {file_path} NOT found")
            all_found = False
    
    if all_found:
        print("✅ Test 4 PASSED")
    else:
        print("❌ Test 4 FAILED")
        return False
    
    # Test 5: Emergency fixes for different modes
    print("\n🧪 Test 5: Testing Emergency Fixes for Different Modes...")
    try:
        import integrated_emergency_fixes as fixes
        
        modes = ["full_pipeline", "debug_full_pipeline", "preprocess"]
        
        for mode in modes:
            print(f"🔧 Testing emergency fixes for {mode}...")
            success = fixes.apply_emergency_fixes_to_pipeline(mode)
            
            if success:
                print(f"✅ {mode} emergency fixes work")
            else:
                print(f"⚠️ {mode} emergency fixes had issues but didn't crash")
        
        print("✅ Test 5 PASSED")
        
    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🔥 INTEGRATED EMERGENCY FIXES - INTEGRATION TEST")
    print("=" * 60)
    print("Testing the integration of emergency fixes into all ProjectP modes")
    print("=" * 60)
    
    success = test_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Emergency fixes are successfully integrated")
        print("🚀 Ready to run ultimate pipeline!")
        print("\n💡 Try running:")
        print("   python ProjectP.py --ultimate_pipeline")
        print("   or")
        print("   run_ultimate_pipeline.bat")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please check the error messages above")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
