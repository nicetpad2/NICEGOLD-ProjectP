#!/usr/bin/env python3
        from enhanced_logging_functions import (
from pathlib import Path
import sys
            import time
        import traceback
"""
Test script for enhanced logging system in ProjectP
"""


# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def test_enhanced_logging():
    """Test the enhanced logging system"""
    try:
        # Test importing enhanced logging functions
            complete_progress_task, 
            display_session_summary, 
            init_session_logger, 
            log_critical, 
            log_error, 
            log_info, 
            log_success, 
            log_warning, 
            progress_context, 
            start_progress_task, 
        )

        print("✅ Enhanced logging functions imported successfully")

        # Initialize logger
        logger = init_session_logger("Test Session")
        print("✅ Logger initialized")

        # Test basic logging
        log_info("Testing info message", "TEST")
        log_success("Testing success message", "TEST")
        log_warning("Testing warning message", "TEST")
        log_error("Testing error message", "TEST")

        # Test progress bars
        with progress_context():
            task_id = "test_task"
            start_progress_task(task_id, "Testing progress...", 100)

            time.sleep(0.5)
            complete_progress_task(task_id, "Test completed")

        # Test session summary
        print("\n" + " = " * 50)
        print("Testing session summary:")
        display_session_summary()

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing Enhanced Logging System")
    print(" = " * 50)

    success = test_enhanced_logging()

    if success:
        print("\n✅ All tests passed! Enhanced logging system is working.")
    else:
        print("\n❌ Tests failed. Check error messages above.")