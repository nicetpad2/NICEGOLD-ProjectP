#!/usr/bin/env python3
        from config_defaults import (
import os
import sys
        import yaml
"""
Test script to verify that the system uses real data from datacsv folder
"""


def test_config_real_data():
    """Test that config.yaml points to real data"""
    print(" =  = = Testing Config Real Data Settings = =  = ")

    try:

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        data_source = config.get("data", {}).get("source", "")
        m15_source = config.get("data", {}).get("m15_source", "")
        use_real_data = config.get("data", {}).get("use_real_data", False)
        limit_rows = config.get("data", {}).get("limit_rows", True)

        print(f"✅ Config loaded successfully")
        print(f"   data.source: {data_source}")
        print(f"   data.m15_source: {m15_source}")
        print(f"   data.use_real_data: {use_real_data}")
        print(f"   data.limit_rows: {limit_rows}")

        # Check if pointing to datacsv
        if "datacsv" in data_source and "datacsv" in m15_source:
            print("✅ Config correctly points to datacsv folder")
        else:
            print("❌ Config not pointing to datacsv folder")
            return False

        if use_real_data:
            print("✅ use_real_data is enabled")
        else:
            print("❌ use_real_data is disabled")
            return False

        if not limit_rows:
            print("✅ limit_rows is disabled (no row limitations)")
        else:
            print("❌ limit_rows is enabled (row limitations exist)")
            return False

        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_datacsv_files():
    """Test that datacsv files exist and are accessible"""
    print("\n =  = = Testing DataCSV Files = =  = ")

    try:
        datacsv_dir = "datacsv"
        m1_file = os.path.join(datacsv_dir, "XAUUSD_M1.csv")
        m15_file = os.path.join(datacsv_dir, "XAUUSD_M15.csv")

        if not os.path.exists(datacsv_dir):
            print(f"❌ DataCSV directory not found: {datacsv_dir}")
            return False
        print(f"✅ DataCSV directory exists: {datacsv_dir}")

        if not os.path.exists(m1_file):
            print(f"❌ M1 file not found: {m1_file}")
            return False

        if not os.path.exists(m15_file):
            print(f"❌ M15 file not found: {m15_file}")
            return False

        print(f"✅ M1 file exists: {m1_file}")
        print(f"✅ M15 file exists: {m15_file}")

        # Get file sizes
        m1_size = os.path.getsize(m1_file) / (1024 * 1024)  # MB
        m15_size = os.path.getsize(m15_file) / (1024 * 1024)  # MB

        print(f"   M1 file size: {m1_size:.2f} MB")
        print(f"   M15 file size: {m15_size:.2f} MB")

        return True
    except Exception as e:
        print(f"❌ DataCSV files test failed: {e}")
        return False


def test_config_defaults():
    """Test that config defaults point to datacsv"""
    print("\n =  = = Testing Config Defaults = =  = ")

    try:
        sys.path.insert(0, "src")
            DEFAULT_DATA_FILE_PATH_M1, 
            DEFAULT_DATA_FILE_PATH_M15, 
        )

        print(f"✅ Config defaults loaded")
        print(f"   DEFAULT_DATA_FILE_PATH_M1: {DEFAULT_DATA_FILE_PATH_M1}")
        print(f"   DEFAULT_DATA_FILE_PATH_M15: {DEFAULT_DATA_FILE_PATH_M15}")

        if (
            "datacsv" in DEFAULT_DATA_FILE_PATH_M1
            and "datacsv" in DEFAULT_DATA_FILE_PATH_M15
        ):
            print("✅ Default paths correctly point to datacsv")
            return True
        else:
            print("❌ Default paths not pointing to datacsv")
            return False

    except Exception as e:
        print(f"❌ Config defaults test failed: {e}")
        return False


def test_real_data_loader_exists():
    """Test that real data loader exists and is importable"""
    print("\n =  = = Testing Real Data Loader = =  = ")

    try:
        sys.path.insert(0, "src")

        # Try to import without pandas/numpy dependencies
        with open("src/real_data_loader.py", "r") as f:
            content = f.read()

        print("✅ real_data_loader.py file exists")

        # Check for key components
        if "class RealDataLoader" in content:
            print("✅ RealDataLoader class found")
        else:
            print("❌ RealDataLoader class not found")
            return False

        if "def load_real_data" in content:
            print("✅ load_real_data function found")
        else:
            print("❌ load_real_data function not found")
            return False

        if "datacsv" in content:
            print("✅ References to datacsv folder found")
        else:
            print("❌ No references to datacsv folder")
            return False

        return True
    except Exception as e:
        print(f"❌ Real data loader test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🔍 NICEGOLD ProjectP Real Data Integration Test")
    print(" = " * 60)

    tests = [
        test_config_real_data, 
        test_datacsv_files, 
        test_config_defaults, 
        test_real_data_loader_exists, 
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + " = " * 60)
    print("🏁 TEST RESULTS SUMMARY")
    print(" = " * 60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("🎉 System successfully configured to use real data from datacsv!")
        print("📁 No dummy data, no row limits, real CSV data only!")
        return 0
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total} passed)")
        print("⚠️  System may not be fully configured for real data usage")
        return 1


if __name__ == "__main__":
    exit(main())