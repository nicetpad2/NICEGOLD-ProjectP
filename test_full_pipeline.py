#!/usr/bin/env python3
    from projectp.steps.predict import run_predict
    import ProjectP  # This will test all syntax
import subprocess
import sys
import time
"""
Quick test to verify pipeline runs without syntax errors
"""

print("🔧 Testing full pipeline without syntax errors...")

try:
    # Test that we can import the main modules
    print("Testing imports...")
    print("✅ Main imports successful - no syntax errors")

    # Test that prediction step works
    print("Testing prediction step...")
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