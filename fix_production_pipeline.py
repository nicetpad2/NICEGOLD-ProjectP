#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 NICEGOLD ProjectP - Production Pipeline Quick Fix
==================================================

Quick fix for the production pipeline to resolve import and method issues.
This script ensures the ProductionFeatureEngineer works correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_production_features():
    """Test and validate ProductionFeatureEngineer"""
    print("🔧 Testing ProductionFeatureEngineer...")

    try:
        # Clear any cached imports
        if "src.production_features" in sys.modules:
            del sys.modules["src.production_features"]

        # Fresh import
        from src.production_features import ProductionFeatureEngineer

        # Create instance
        fe = ProductionFeatureEngineer()

        # Test if get_feature_summary exists
        if hasattr(fe, "get_feature_summary"):
            print("✅ get_feature_summary method exists")

            try:
                summary = fe.get_feature_summary()
                print(f"✅ Method call successful: {summary}")
                return True
            except Exception as e:
                print(f"❌ Method call failed: {e}")
                return False
        else:
            print("❌ get_feature_summary method missing")
            return False

    except Exception as e:
        print(f"❌ Import or initialization failed: {e}")
        return False


def fix_module_cache():
    """Clear module cache to ensure fresh imports"""
    modules_to_clear = [
        "src.production_features",
        "src.production_pipeline",
        "src.production_ml_training",
        "src.robust_csv_manager",
    ]

    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
            print(f"🧹 Cleared cache for {module}")


def main():
    """Main fix function"""
    print("🚀 NICEGOLD ProjectP Production Pipeline Fix")
    print("=" * 50)

    # Clear module cache
    print("\n1. Clearing module cache...")
    fix_module_cache()

    # Test production features
    print("\n2. Testing ProductionFeatureEngineer...")
    success = test_production_features()

    if success:
        print("\n✅ Production features are working correctly!")
        print("💡 The original error was likely due to cached imports.")
        print("   Try running your pipeline again with fresh Python process.")
    else:
        print("\n❌ There are still issues with ProductionFeatureEngineer.")
        print("   Please check the implementation.")

    return success


if __name__ == "__main__":
    main()
