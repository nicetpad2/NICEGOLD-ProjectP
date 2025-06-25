#!/usr/bin/env python3
from datetime import datetime
            from ProjectP import run_full_pipeline, main
            import json
import os
                import runpy
import sys
import time
import warnings
"""
Safe ProjectP Launcher - Prevents infinite loops and import errors
"""


# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, '.')

print("Safe ProjectP Launcher")
print(" = " * 40)
print(f"Start time: {datetime.now()}")

# Load fallbacks first
print("Loading fallbacks...")
try:
    exec(open('pydantic_fallback.py').read())
    print("Pydantic fallback loaded")
except Exception as e:
    print(f"Pydantic fallback error: {e}")

try:
    exec(open('sklearn_fallback.py').read())
    print("Sklearn fallback loaded")
except Exception as e:
    print(f"Sklearn fallback error: {e}")

def run_projectp_safe():
    """Run ProjectP with safety measures"""
    start_time = time.time()

    try:
        print("\nLoading ProjectP...")

        # Import ProjectP components safely
        try:
            print("ProjectP imported successfully")
        except Exception as e:
            print(f"ProjectP import error: {e}")
            print("Trying alternative import...")

            # Try direct execution
            try:
                result = runpy.run_path('ProjectP.py', run_name = '__main__')
                print("ProjectP executed via runpy")
                return True
            except Exception as e2:
                print(f"Alternative import failed: {e2}")
                return False

        # Run the pipeline
        print("\nRunning full pipeline...")

        if ' -  - run_full_pipeline' in sys.argv:
            result = run_full_pipeline()
        else:
            result = main()

        runtime = time.time() - start_time
        print(f"\nPipeline completed in {runtime:.1f} seconds")

        # Check for results
        if os.path.exists('classification_report.json'):
            with open('classification_report.json', 'r') as f:
                data = json.load(f)
            accuracy = data.get('accuracy', 0)
            print(f"Final Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

        return True

    except KeyboardInterrupt:
        print("\nStopped by user")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False

if __name__ == "__main__":
    success = run_projectp_safe()
    if success:
        print("\nSafe execution completed!")
    else:
        print("\nExecution had issues - check logs")