#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for NICEGOLD Modern Logger
Shows off all the beautiful features

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

def main():
    print("🚀 NICEGOLD Modern Logger Demo")
    print("=" * 50)
    
    try:
        from utils.modern_logger import (
            NotificationType,
            critical,
            error,
            get_logger,
            info,
            setup_logger,
            success,
            warning,
        )

        # Setup logger
        logger = setup_logger("DEMO", enable_file_logging=True)
        
        print("\n1. 📝 Testing Basic Logging...")
        info("This is an info message")
        success("This is a success message")
        warning("This is a warning message")
        error("This is an error message")
        
        print("\n2. 📊 Testing Progress Bar...")
        with logger.progress_bar("Processing demo data", total=20) as update:
            for i in range(20):
                time.sleep(0.05)
                update()
        
        print("\n3. ⏳ Testing Status Spinner...")
        with logger.status("Loading configuration...", spinner="dots"):
            time.sleep(2)
        
        print("\n4. 🔔 Testing Notifications...")
        logger.notify(
            "Demo completed successfully!", 
            NotificationType.SUCCESS,
            title="NICEGOLD Demo"
        )
        
        print("\n5. 📋 Testing Table Display...")
        demo_data = [
            {"Feature": "Rich Terminal Output", "Status": "✅ Working", "Performance": "Excellent"},
            {"Feature": "Progress Bars", "Status": "✅ Working", "Performance": "Fast"},
            {"Feature": "Notifications", "Status": "✅ Working", "Performance": "Smooth"},
            {"Feature": "Error Handling", "Status": "✅ Working", "Performance": "Robust"},
        ]
        logger.display_table(demo_data, title="🎯 Modern Logger Features")
        
        print("\n6. 🏗️ Testing Tree Display...")
        tree_data = {
            "NICEGOLD ProjectP": {
                "Core Features": {
                    "Modern Logger": ["Rich formatting", "Progress bars", "Notifications"],
                    "Trading System": ["ML Models", "Risk Management", "Backtesting"],
                    "AI Agents": ["Auto-optimization", "Error correction", "Performance tuning"]
                },
                "Infrastructure": {
                    "Monitoring": ["Health checks", "Performance metrics", "Alerts"],
                    "Security": ["Input validation", "Error handling", "Audit logs"]
                }
            }
        }
        logger.display_tree(tree_data, title="🏗️ System Architecture")
        
        print("\n7. ⏱️ Testing Performance Tracking...")
        with logger.timer("Demo calculation"):
            # Simulate some work
            result = sum(i**2 for i in range(100000))
            time.sleep(0.5)
        
        print("\n8. 💻 Testing System Information...")
        logger.display_system_info()
        
        print("\n9. 📈 Testing Session Summary...")
        logger.display_summary()
        
        success("🎉 Modern Logger Demo completed successfully!")
        success("All features are working perfectly!")
        
        # Export logs
        exported_file = logger.export_logs()
        info(f"📁 Demo logs exported to: {exported_file}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
