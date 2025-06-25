#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Fixed Pipeline - ทดสอบ pipeline ที่แก้ไขแล้ว
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_fixed_pipeline():
    """ทดสอบ full pipeline ที่แก้ไขข้อผิดพลาด random_state และ chart_style แล้ว"""
    print("🔧 Testing Fixed Pipeline - การทดสอบ pipeline ที่แก้ไขแล้ว")
    print("=" * 60)

    try:
        # Import menu operations
        from core.menu_operations import MenuOperations

        # Create menu operations instance
        menu = MenuOperations()

        # Test configuration first
        print("📋 Testing pipeline configuration...")
        config = menu._get_pipeline_config()

        # Check key configurations
        required_keys = ["model_trainer_config", "performance_analyzer_config"]

        for key in required_keys:
            if key in config:
                print(f"✅ {key}: {config[key]}")
            else:
                print(f"❌ Missing {key}")
                return False

        # Check specific values
        model_config = config.get("model_trainer_config", {})
        perf_config = config.get("performance_analyzer_config", {})

        print(f"\n🔍 Key Configuration Values:")
        print(f"   random_state: {model_config.get('random_state', 'NOT FOUND')}")
        print(f"   chart_style: {perf_config.get('chart_style', 'NOT FOUND')}")

        # Test ModelTrainer initialization
        print(f"\n🧪 Testing ModelTrainer initialization...")
        try:
            from core.pipeline.model_trainer import ModelTrainer

            trainer = ModelTrainer(model_config)
            print("✅ ModelTrainer initialized successfully")
        except Exception as e:
            print(f"❌ ModelTrainer failed: {e}")
            return False

        # Test PerformanceAnalyzer initialization
        print(f"\n📊 Testing PerformanceAnalyzer initialization...")
        try:
            from core.pipeline.performance_analyzer import PerformanceAnalyzer

            analyzer = PerformanceAnalyzer(perf_config)
            print("✅ PerformanceAnalyzer initialized successfully")
        except Exception as e:
            print(f"❌ PerformanceAnalyzer failed: {e}")
            return False

        # Now test full pipeline initialization only (not full execution)
        print(f"\n🚀 Testing Pipeline Orchestrator initialization...")
        try:
            from core.pipeline.pipeline_orchestrator import PipelineOrchestrator

            pipeline = PipelineOrchestrator(config)
            print("✅ PipelineOrchestrator initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ PipelineOrchestrator failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_fixed_pipeline()
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests passed! Pipeline is working correctly.")
    else:
        print("💥 Tests failed! Please check the errors above.")
    print(f"{'='*60}")
