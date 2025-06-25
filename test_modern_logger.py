#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for NICEGOLD Modern Logger
Demonstrates all features of the modern logging system

Author: NICEGOLD Enterprise
Version: 1.0
Date: June 25, 2025
"""

import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_logging():
    """Test basic logging functionality"""
    try:
        from utils.modern_logger import (
            LogLevel,
            NotificationType,
            critical,
            debug,
            error,
            info,
            progress,
            setup_logger,
            success,
            warning,
        )

        # Setup logger
        logger = setup_logger("TEST_LOGGER", level=LogLevel.DEBUG)
        
        print("🧪 Testing Basic Logging Functions...")
        print("=" * 50)
        
        # Test all log levels
        debug("This is a debug message")
        info("This is an info message")
        success("This is a success message")
        warning("This is a warning message")
        error("This is an error message")
        critical("This is a critical message")
        progress("This is a progress message")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic logging test failed: {e}")
        return False


def test_progress_bars():
    """Test progress bar functionality"""
    try:
        from utils.modern_logger import get_logger
        
        logger = get_logger()
        print("\n🧪 Testing Progress Bars...")
        print("=" * 50)
        
        # Test progress bar with known total
        with logger.progress_bar("Processing items", total=100) as update:
            for i in range(100):
                time.sleep(0.01)  # Simulate work
                update()
        
        # Test progress bar with unknown total
        with logger.progress_bar("Loading data") as update:
            for i in range(50):
                time.sleep(0.02)
                update()
        
        return True
        
    except Exception as e:
        print(f"❌ Progress bar test failed: {e}")
        return False


def test_status_spinner():
    """Test status spinner functionality"""
    try:
        from utils.modern_logger import get_logger
        
        logger = get_logger()
        print("\n🧪 Testing Status Spinner...")
        print("=" * 50)
        
        # Test different spinners
        spinners = ["dots", "line", "pipe", "dots2", "dots3"]
        
        for spinner in spinners:
            with logger.status(f"Testing {spinner} spinner", spinner=spinner):
                time.sleep(1)
        
        return True
        
    except Exception as e:
        print(f"❌ Status spinner test failed: {e}")
        return False


def test_notifications():
    """Test notification functionality"""
    try:
        from utils.modern_logger import NotificationType, get_logger
        
        logger = get_logger()
        print("\n🧪 Testing Notifications...")
        print("=" * 50)
        
        # Test different notification types
        logger.notify("This is an info notification", NotificationType.INFO)
        time.sleep(0.5)
        
        logger.notify("This is a success notification", NotificationType.SUCCESS)
        time.sleep(0.5)
        
        logger.notify("This is a warning notification", NotificationType.WARNING)
        time.sleep(0.5)
        
        logger.notify("This is an error notification", NotificationType.ERROR)
        time.sleep(0.5)
        
        return True
        
    except Exception as e:
        print(f"❌ Notification test failed: {e}")
        return False


def test_tables_and_trees():
    """Test table and tree display functionality"""
    try:
        from utils.modern_logger import get_logger
        
        logger = get_logger()
        print("\n🧪 Testing Tables and Trees...")
        print("=" * 50)
        
        # Test table display
        test_data = [
            {"Name": "Bitcoin", "Price": "$45,000", "Change": "+5.2%"},
            {"Name": "Ethereum", "Price": "$3,200", "Change": "-2.1%"},
            {"Name": "Gold", "Price": "$1,850", "Change": "+0.8%"}
        ]
        
        logger.display_table(test_data, title="💰 Market Data")
        
        # Test tree display
        tree_data = {
            "Trading System": {
                "Data Sources": ["Binance", "Coinbase", "Kraken"],
                "Strategies": {
                    "Technical Analysis": ["RSI", "MACD", "Bollinger Bands"],
                    "Machine Learning": ["Random Forest", "XGBoost", "Neural Networks"]
                },
                "Risk Management": ["Stop Loss", "Position Sizing", "Portfolio Diversification"]
            }
        }
        
        logger.display_tree(tree_data, title="🏗️ System Architecture")
        
        return True
        
    except Exception as e:
        print(f"❌ Table/Tree test failed: {e}")
        return False


def test_error_handling():
    """Test error handling functionality"""
    try:
        from utils.modern_logger import get_logger
        
        logger = get_logger()
        print("\n🧪 Testing Error Handling...")
        print("=" * 50)
        
        # Test exception handling
        try:
            # Deliberately cause an error
            result = 1 / 0
        except Exception as e:
            logger.handle_exception(e, "Division test", fatal=False)
        
        # Test custom error logging
        logger.error("Custom error message", exception=ValueError("Custom exception"))
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_performance_tracking():
    """Test performance tracking functionality"""
    try:
        from utils.modern_logger import get_logger
        
        logger = get_logger()
        print("\n🧪 Testing Performance Tracking...")
        print("=" * 50)
        
        # Test timer context manager
        with logger.timer("Data processing simulation"):
            time.sleep(1)
            # Simulate some work
            data = [i ** 2 for i in range(10000)]
        
        # Test named timers
        logger.start_timer("manual_timer")
        time.sleep(0.5)
        duration = logger.end_timer("manual_timer")
        
        print(f"Manual timer duration: {duration:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance tracking test failed: {e}")
        return False


def test_system_info():
    """Test system information display"""
    try:
        from utils.modern_logger import get_logger
        
        logger = get_logger()
        print("\n🧪 Testing System Information...")
        print("=" * 50)
        
        logger.display_system_info()
        
        return True
        
    except Exception as e:
        print(f"❌ System info test failed: {e}")
        return False


def test_session_summary():
    """Test session summary functionality"""
    try:
        from utils.modern_logger import get_logger
        
        logger = get_logger()
        print("\n🧪 Testing Session Summary...")
        print("=" * 50)
        
        # Generate some activity
        logger.info("Test activity 1")
        logger.warning("Test warning")
        logger.error("Test error")
        logger.success("Test success")
        
        # Display summary
        logger.display_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Session summary test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 NICEGOLD Modern Logger Test Suite")
    print("=" * 50)
    print("Testing modern logging system with all features...")
    print()
    
    tests = [
        ("Basic Logging", test_basic_logging),
        ("Progress Bars", test_progress_bars),
        ("Status Spinner", test_status_spinner),
        ("Notifications", test_notifications),
        ("Tables and Trees", test_tables_and_trees),
        ("Error Handling", test_error_handling),
        ("Performance Tracking", test_performance_tracking),
        ("System Information", test_system_info),
        ("Session Summary", test_session_summary),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                print(f"✅ {test_name} test passed!")
                passed += 1
            else:
                print(f"❌ {test_name} test failed!")
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Modern logger is working perfectly!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    # Test log export
    try:
        from utils.modern_logger import get_logger
        logger = get_logger()
        exported_file = logger.export_logs()
        print(f"📁 Test logs exported to: {exported_file}")
    except Exception as e:
        print(f"⚠️ Log export failed: {e}")


if __name__ == "__main__":
    main()
