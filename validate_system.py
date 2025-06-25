#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete System Validation Test
ทดสอบระบบอย่างครอบคลุม
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_dependencies():
    """Test required dependencies"""
    print("🔍 Testing Dependencies...")

    required_packages = [
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "joblib",
        "yaml",
        "tqdm",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == "yaml":
                import yaml
            elif package == "sklearn":
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    return len(missing_packages) == 0


def test_core_modules():
    """Test core system modules"""
    print("\n🔧 Testing Core Modules...")

    try:
        from core.config import get_config
        from core.menu_operations import MenuOperations
        from core.system import get_system

        print("✅ Core modules imported successfully")

        # Test config
        config = get_config()
        print("✅ Configuration loaded")

        # Test menu operations
        menu = MenuOperations()
        print("✅ MenuOperations initialized")

        # Test system
        system = get_system()
        print("✅ System utilities loaded")

        return True

    except Exception as e:
        print(f"❌ Core modules failed: {e}")
        return False


def test_pipeline_components():
    """Test pipeline components"""
    print("\n🚀 Testing Pipeline Components...")

    try:
        from core.pipeline.model_trainer import ModelTrainer
        from core.pipeline.performance_analyzer import PerformanceAnalyzer
        from core.pipeline.pipeline_orchestrator import PipelineOrchestrator

        # Test configurations
        model_config = {
            "random_state": 42,
            "test_size": 0.2,
            "models_to_train": ["random_forest"],
            "cv_folds": 3,
            "save_models": False,
        }

        perf_config = {
            "chart_style": "seaborn-v0_8",
            "save_charts": False,
            "figure_size": (10, 6),
        }

        # Test ModelTrainer
        trainer = ModelTrainer(model_config)
        print("✅ ModelTrainer initialized")

        # Test PerformanceAnalyzer
        analyzer = PerformanceAnalyzer(perf_config)
        print("✅ PerformanceAnalyzer initialized")

        # Test basic pipeline config
        from core.menu_operations import MenuOperations

        menu = MenuOperations()
        pipeline_config = menu._get_pipeline_config()
        print("✅ Pipeline configuration generated")

        return True

    except Exception as e:
        print(f"❌ Pipeline components failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_sources():
    """Test data sources"""
    print("\n📊 Testing Data Sources...")

    datacsv_path = Path("datacsv")
    if not datacsv_path.exists():
        print("❌ datacsv folder not found")
        return False

    csv_files = list(datacsv_path.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in datacsv folder")
        return False

    print(f"✅ Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        file_size = csv_file.stat().st_size / (1024 * 1024)  # MB
        print(f"   📄 {csv_file.name} ({file_size:.1f} MB)")

    # Test loading a CSV file
    try:
        import pandas as pd

        test_file = csv_files[0]
        df = pd.read_csv(test_file, nrows=5)
        print(f"✅ Successfully loaded sample from {test_file.name}")
        print(f"   Columns: {list(df.columns)}")

        return True

    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return False


def test_system_integration():
    """Test full system integration"""
    print("\n🎯 Testing System Integration...")

    try:
        from core.menu_operations import MenuOperations

        menu = MenuOperations()

        # Test data source selection
        data_source = menu._get_data_source()
        if data_source:
            print(f"✅ Data source found: {Path(data_source).name}")
        else:
            print("❌ No valid data source found")
            return False

        # Test pipeline config
        config = menu._get_pipeline_config()
        required_keys = [
            "model_trainer_config",
            "performance_analyzer_config",
            "data_validator_config",
        ]

        for key in required_keys:
            if key in config:
                print(f"✅ {key} present in configuration")
            else:
                print(f"❌ {key} missing from configuration")
                return False

        print("✅ System integration test passed")
        return True

    except Exception as e:
        print(f"❌ System integration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 NICEGOLD ProjectP - Complete System Validation")
    print("=" * 60)

    tests = [
        ("Dependencies", test_dependencies),
        ("Core Modules", test_core_modules),
        ("Pipeline Components", test_pipeline_components),
        ("Data Sources", test_data_sources),
        ("System Integration", test_system_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print(f"\n{'=' * 60}")
    print(f"🏆 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for production.")
        print("\n🚀 You can now run the full pipeline:")
        print("   python3 ProjectP.py")
        print("   Then select option 1 (Full Pipeline)")
        return True
    else:
        print("💥 Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
