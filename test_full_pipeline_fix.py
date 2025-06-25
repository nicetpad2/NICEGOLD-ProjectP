#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Full Pipeline (Option 1) to verify the random_state fix
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_full_pipeline():
    """Test the full pipeline to verify random_state fix"""
    print("🧪 Testing Full Pipeline (Option 1)...")
    print("=" * 60)

    try:
        # Import menu operations
        from core.menu_operations import MenuOperations

        # Initialize
        menu_ops = MenuOperations()

        print("✅ MenuOperations initialized successfully")

        # Test pipeline config
        pipeline_config = menu_ops._get_pipeline_config()
        print(f"✅ Pipeline config created")

        # Check if random_state is in model_trainer_config
        model_config = pipeline_config.get("model_trainer_config", {})
        random_state = model_config.get("random_state")

        if random_state is not None:
            print(f"✅ random_state found: {random_state}")
        else:
            print("❌ random_state NOT found in model_trainer_config")
            return False

        # Test data source
        data_source = menu_ops._get_data_source()
        if data_source:
            print(f"✅ Data source found: {data_source}")
        else:
            print("⚠️ No data source found (will use datacsv)")

        # Try to run full pipeline
        print("\n🚀 Attempting to run full pipeline...")
        result = menu_ops.full_pipeline()

        if result:
            print("✅ Full pipeline completed successfully!")
        else:
            print("❌ Full pipeline failed")

        return result

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
