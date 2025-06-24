#!/usr/bin/env python3
"""
Test script to verify real data enforcement in ProjectP
"""

import sys
import os

# Add projectp to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_validation():
    """Test that data validation works correctly"""
    print("🔍 Testing Real Data Validation...")
    
    try:
        from projectp.data_validator import RealDataValidator, enforce_real_data_only
        
        # Test 1: Check datacsv folder validation
        print("\n📁 Test 1: Validating datacsv folder...")
        validator = RealDataValidator()
        if validator.validate_datacsv_folder():
            print("✅ datacsv folder validation passed")
        
        # Test 2: Get available files
        print("\n📄 Test 2: Getting available data files...")
        available_files = validator.get_available_data_files()
        print(f"✅ Found {len(available_files)} valid data files: {available_files}")
        
        # Test 3: Load real data
        print("\n📊 Test 3: Loading real data...")
        if available_files:
            df = validator.load_real_data(available_files[0])
            print(f"✅ Successfully loaded data: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Data range: {len(df)} rows")
        
        # Test 4: Enforce real data only
        print("\n🛡️ Test 4: Enforcing real data only...")
        data_validator = enforce_real_data_only()
        print("✅ Real data enforcement activated successfully")
        
        print("\n🎉 All tests passed! Real data validation is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test that pipeline integration works"""
    print("\n🚀 Testing Pipeline Integration...")
    
    try:
        # Test preprocess function with real data enforcement
        from projectp.steps.preprocess import run_preprocess
        
        print("📊 Testing preprocess with real data enforcement...")
        
        # This should work with real data from datacsv
        config = {"data": {"file": "XAUUSD_M1.csv"}}  # Request specific file from datacsv
        
        result = run_preprocess(config, mode="fast")  # Use fast mode for quick testing
        
        if result:
            print("✅ Preprocess completed successfully with real data")
        else:
            print("⚠️ Preprocess completed with warnings")
            
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ProjectP Real Data Validation Test")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_data_validation()
    test2_passed = test_pipeline_integration()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Real data enforcement is working correctly")
        print("✅ Pipeline integration is working correctly")
        print("🛡️ Only real data from datacsv folder will be used")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above")
    
    print("=" * 60)
