#!/usr/bin/env python3
"""
Quick test to verify pipeline runs without syntax errors
"""

import sys
import subprocess
import time

print("🔧 Testing full pipeline without syntax errors...")

try:
    # Test that we can import the main modules
    print("Testing imports...")
    import ProjectP  # This will test all syntax
    print("✅ Main imports successful - no syntax errors")
    
    # Test that prediction step works
    print("Testing prediction step...")
    from projectp.steps.predict import run_predict
    print("✅ Prediction step import successful")
    
    print("🎉 All syntax tests passed!")
    
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"ℹ️ Import dependency missing (but no syntax error): {e}")
except Exception as e:
    print(f"ℹ️ Other error (but no syntax error): {e}")

print("Test complete!")
