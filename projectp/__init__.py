# ProjectP package init
# Core pipeline imports with fallbacks

import subprocess
import sys

try:
    from .dashboard import main as dashboard_main
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    dashboard_main = None

try:
    from .pipeline import (
        run_debug_full_pipeline,
        run_full_pipeline,
        run_ultimate_pipeline,
    )
    PIPELINE_AVAILABLE = True
    print("✅ Pipeline functions imported successfully")
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"⚠️ Pipeline functions not available: {e}")
    
    # Create placeholder functions
    def run_full_pipeline(*args, **kwargs):
        print("⚠️ Pipeline not available - using fallback")
        return {}
    
    def run_debug_full_pipeline(*args, **kwargs):
        print("⚠️ Debug pipeline not available - using fallback")
        return {}
    
    def run_ultimate_pipeline(*args, **kwargs):
        print("⚠️ Ultimate pipeline not available - using fallback")
        return {}

def run_dashboard():
    """Run dashboard if available"""
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'projectp/dashboard.py'])
    except Exception as e:
        print(f"⚠️ Dashboard not available: {e}")

__all__ = ['run_full_pipeline', 'run_debug_full_pipeline', 'run_ultimate_pipeline', 'run_dashboard']
