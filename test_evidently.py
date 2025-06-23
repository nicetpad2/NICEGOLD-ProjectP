#!/usr/bin/env python3
"""Test script to check evidently import and typing compatibility."""

import sys
print(f"Python version: {sys.version}")

try:
    import evidently
    print(f"✅ Evidently imported successfully")
    print(f"Evidently version: {evidently.__version__}")
    
    # Test the specific import that was causing issues
    from evidently.metric_preset import DataQualityPreset
    print("✅ DataQualityPreset imported successfully")
    
    from evidently.report import Report
    print("✅ Report imported successfully")
    
    print("🎉 All evidently imports working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
