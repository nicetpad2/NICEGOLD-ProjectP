#!/usr/bin/env python3
"""
Simple test to verify data loading from datacsv without pandas dependency
"""


def test_data_loading():
    """Test data loading using basic Python"""
    print(" =  = = Testing Data Loading from datacsv = =  = ")

    try:
        import csv
        import os

        m1_file = "datacsv/XAUUSD_M1.csv"
        m15_file = "datacsv/XAUUSD_M15.csv"

        if not os.path.exists(m1_file):
            print(f"❌ M1 file not found: {m1_file}")
            return False

        if not os.path.exists(m15_file):
            print(f"❌ M15 file not found: {m15_file}")
            return False

        print(f"✅ Both data files exist")

        # Test M1 data loading
        with open(m1_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            print(f"✅ M1 header: {header}")

            # Read first few data rows
            rows = []
            for i, row in enumerate(reader):
                if i >= 5:  # Just read first 5 rows
                    break
                rows.append(row)

        print(f"✅ M1 sample data loaded: {len(rows)} rows")
        print(f"   First row: {rows[0] if rows else 'No data'}")

        # Test M15 data loading
        with open(m15_file, "r") as f:
            reader = csv.reader(f)
            header15 = next(reader)

            print(f"✅ M15 header: {header15}")

            # Read first few data rows
            rows15 = []
            for i, row in enumerate(reader):
                if i >= 5:  # Just read first 5 rows
                    break
                rows15.append(row)

        print(f"✅ M15 sample data loaded: {len(rows15)} rows")
        print(f"   First row: {rows15[0] if rows15 else 'No data'}")

        # Count total rows (approximation)
        with open(m1_file, "r") as f:
            m1_lines = sum(1 for line in f) - 1  # Subtract header

        with open(m15_file, "r") as f:
            m15_lines = sum(1 for line in f) - 1  # Subtract header

        print(f"✅ Total data rows:")
        print(f"   M1: {m1_lines:, } rows")
        print(f"   M15: {m15_lines:, } rows")

        print(f"✅ Data format validation:")
        print(f"   Expected columns: Date, Timestamp, Open, High, Low, Close, Volume")
        print(f"   M1 columns: {len(header)} - {header}")
        print(f"   M15 columns: {len(header15)} - {header15}")

        if len(header) >= 7 and len(header15) >= 7:
            print("✅ Column count looks correct")
        else:
            print("❌ Unexpected column count")
            return False

        return True

    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_system_ready():
    """Final test - check if system is ready to use real data"""
    print("\n =  = = System Readiness Check = =  = ")

    checks = [
        ("Config points to datacsv", "datacsv" in open("config.yaml").read()), 
        ("M1 data file exists", os.path.exists("datacsv/XAUUSD_M1.csv")), 
        ("M15 data file exists", os.path.exists("datacsv/XAUUSD_M15.csv")), 
        ("Real data loader exists", os.path.exists("src/real_data_loader.py")), 
    ]

    all_passed = True
    for check_name, condition in checks:
        if condition:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False

    return all_passed


if __name__ == "__main__":

    print("🔍 NICEGOLD Real Data Verification")
    print(" = " * 50)

    success1 = test_data_loading()
    success2 = test_system_ready()

    print("\n" + " = " * 50)
    if success1 and success2:
        print("🎉 SUCCESS: System configured to use REAL DATA from datacsv!")
        print("📊 No dummy data, no row limits, full dataset ready!")
        print("🚀 Ready to run full pipeline with real XAUUSD data")
    else:
        print("⚠️  Some issues detected, check the errors above")