#!/usr/bin/env python3
"""
Test script to verify Rich live display conflict resolution
"""

import os
import sys
import threading
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modern_logger import get_logger


def test_concurrent_displays():
    """Test concurrent progress bars and status displays"""
    logger = get_logger("TEST")
    
    print("🧪 Testing Rich live display conflict resolution...")
    
    # Test 1: Nested status calls (should fall back gracefully)
    print("\n1️⃣ Testing nested status displays...")
    with logger.status("Outer status"):
        time.sleep(1)
        logger.simple_status("Inner status (should not conflict)")
        time.sleep(1)
    
    # Test 2: Progress bar while status is active
    print("\n2️⃣ Testing progress bar with active status...")
    with logger.status("Status active"):
        time.sleep(0.5)
        logger.show_progress("Progress during status", 1, 3)
        time.sleep(0.5)
        logger.show_progress("Progress during status", 2, 3)
        time.sleep(0.5)
        logger.show_progress("Progress during status", 3, 3)
    
    # Test 3: Proper sequential use
    print("\n3️⃣ Testing proper sequential use...")
    with logger.progress_bar("First operation", total=3) as update:
        for i in range(3):
            time.sleep(0.5)
            update()
    
    with logger.status("Second operation"):
        time.sleep(1)
    
    # Test 4: Threaded concurrent access
    print("\n4️⃣ Testing threaded concurrent access...")
    
    def worker1():
        logger.simple_status("Worker 1 starting")
        time.sleep(1)
        logger.show_progress("Worker 1 progress", 50, 100)
        logger.success("Worker 1 complete")
    
    def worker2():
        time.sleep(0.5)
        logger.simple_status("Worker 2 starting")
        time.sleep(1)
        logger.show_progress("Worker 2 progress", 75, 100)
        logger.success("Worker 2 complete")
    
    thread1 = threading.Thread(target=worker1)
    thread2 = threading.Thread(target=worker2)
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    # Test 5: Mixed rich and simple logging
    print("\n5️⃣ Testing mixed rich and simple logging...")
    logger.info("Regular info message")
    logger.show_progress("Safe progress", 1, 2)
    logger.warning("Warning message")
    logger.show_progress("Safe progress", 2, 2)
    logger.success("All tests completed!")
    
    # Show session summary
    print("\n📊 Session Summary:")
    logger.show_session_summary()

if __name__ == "__main__":
    test_concurrent_displays()
