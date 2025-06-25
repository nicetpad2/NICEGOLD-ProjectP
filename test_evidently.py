#!/usr/bin/env python3
    from evidently.metric_preset import DataQualityPreset
    from evidently.report import Report
    import evidently
import sys
"""Test script to check evidently import and typing compatibility."""

print(f"Python version: {sys.version}")

try:
    print(f"✅ Evidently imported successfully")
    print(f"Evidently version: {evidently.__version__}")

    # Test the specific import that was causing issues
    print("✅ DataQualityPreset imported successfully")

    print("✅ Report imported successfully")

    print("🎉 All evidently imports working correctly!")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")