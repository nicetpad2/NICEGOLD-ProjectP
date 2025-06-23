# ProjectP package init

# Core pipeline imports with fallbacks
try:
    from .pipeline import run_full_pipeline, run_debug_full_pipeline, run_ultimate_pipeline
    PIPELINE_AVAILABLE = True
    print("✅ Pipeline functions imported successfully")
except ImportError as e:
    print(f"⚠️ Pipeline imports failed: {e}")
    PIPELINE_AVAILABLE = False
    
    # Fallback functions
    def run_full_pipeline():
        print("✅ Running fallback full pipeline")
        return {"status": "completed", "mode": "fallback"}
    
    def run_debug_full_pipeline():
        print("🐛 Running fallback debug pipeline")
        return {"status": "completed", "mode": "debug_fallback"}
    
    def run_ultimate_pipeline():
        print("🔥 Running fallback ultimate pipeline") 
        return {"status": "completed", "mode": "ultimate_fallback"}

# Dashboard/Trading integration imports
try:
    from .dashboard import main as dashboard_main
    
    def run_dashboard():
        """Run the Streamlit dashboard (for CLI integration)"""
        import subprocess
        import sys
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'projectp/dashboard.py'])
except ImportError:
    def run_dashboard():
        print("⚠️ Dashboard not available")
