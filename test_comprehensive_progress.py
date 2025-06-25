#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Test Comprehensive Progress System
"""

print("🔍 Testing comprehensive progress system...")

try:
    from comprehensive_full_pipeline_progress import ComprehensiveProgressSystem
    print("✅ ComprehensiveProgressSystem imported successfully")
    
    progress_system = ComprehensiveProgressSystem()
    print("✅ ComprehensiveProgressSystem initialized successfully")
    
    # Test quick run
    print("🚀 Running test pipeline...")
    results = progress_system.run_full_pipeline_with_complete_progress()
    print(f"✅ Test completed with status: {results['pipeline_status']}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Runtime error: {e}")
    import traceback
    traceback.print_exc()

print("🏁 Test completed!")
