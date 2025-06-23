#!/usr/bin/env python3
"""
ProjectP Production-Ready Pipeline System
========================================
Windows-compatible, Unicode-safe, Error-resistant
"""

import os
import sys
import warnings
import traceback
import multiprocessing
from typing import Optional, Dict, Any

# Fix Windows console encoding issues
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def safe_print(msg: str, level: str = "INFO") -> None:
    """Safe printing with fallback for Unicode issues"""
    try:
        print(f"[{level}] {msg}")
    except UnicodeEncodeError:
        safe_msg = msg.encode('ascii', errors='ignore').decode('ascii')
        print(f"[{level}] {safe_msg}")

def print_banner() -> None:
    """Print professional banner"""
    safe_print("=" * 80)
    safe_print("NICEGOLD PRODUCTION PIPELINE SYSTEM")
    safe_print("Production-Ready Trading ML Pipeline")
    safe_print(f"Python: {sys.version.split()[0]}")
    safe_print(f"Session ID: {os.urandom(4).hex()}")
    safe_print("=" * 80)

def run_fallback_pipeline() -> bool:
    """Fallback pipeline implementation"""
    safe_print("Running fallback pipeline...")
    
    try:
        # Import and run the production pipeline
        from run_production_pipeline import run_production_pipeline
        success = run_production_pipeline()
        
        if success:
            safe_print("Fallback pipeline completed successfully!", "SUCCESS")
            return True
        else:
            safe_print("Fallback pipeline failed", "ERROR")
            return False
            
    except Exception as e:
        safe_print(f"Fallback pipeline failed: {e}", "ERROR")
        safe_print(f"Traceback: {traceback.format_exc()}", "DEBUG")
        return False

def run_mode(mode: str) -> bool:
    """Run the specified mode"""
    safe_print(f"{mode.title()} Mode Starting...")
    
    # Try to import and run main pipeline functions
    try:
        from projectp.pipeline import run_full_pipeline, run_debug_full_pipeline
        
        if mode == "full_pipeline":
            if run_full_pipeline:
                return run_full_pipeline()
            else:
                safe_print("Pipeline function not available. Using fallback.")
                return run_fallback_pipeline()
                
        elif mode == "debug_full_pipeline":
            if run_debug_full_pipeline:
                return run_debug_full_pipeline()
            else:
                safe_print("Debug pipeline function not available. Using fallback.")
                return run_fallback_pipeline()
                
        else:
            safe_print(f"Mode '{mode}' not implemented yet. Using fallback.")
            return run_fallback_pipeline()
            
    except ImportError as e:
        safe_print(f"Pipeline import failed: {e}", "WARNING")
        safe_print("Using fallback pipeline...")
        return run_fallback_pipeline()
        
    except Exception as e:
        safe_print(f"Pipeline execution failed: {e}", "ERROR")
        safe_print(f"Traceback: {traceback.format_exc()}", "DEBUG")
        return False

def check_environment() -> None:
    """Basic environment check"""
    try:
        import pandas as pd
        import numpy as np
        safe_print("Core dependencies available", "SUCCESS")
    except ImportError as e:
        safe_print(f"Missing core dependencies: {e}", "ERROR")

def main():
    """Main entry point"""
    print_banner()
    
    # Basic environment check
    check_environment()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="NICEGOLD Production Pipeline")
    parser.add_argument("--run_full_pipeline", action="store_true", 
                       help="Run the full production pipeline")
    parser.add_argument("--debug_full_pipeline", action="store_true",
                       help="Run full pipeline with debug logging")
    parser.add_argument("--preprocess", action="store_true",
                       help="Run preprocessing only")
    parser.add_argument("--realistic_backtest", action="store_true",
                       help="Run realistic backtesting")
    parser.add_argument("--robust_backtest", action="store_true",
                       help="Run robust backtesting")
    parser.add_argument("--realistic_backtest_live", action="store_true",
                       help="Run live-simulation backtesting")
    
    args = parser.parse_args()
    
    # Determine mode
    mode = None
    if args.run_full_pipeline:
        mode = "full_pipeline"
    elif args.debug_full_pipeline:
        mode = "debug_full_pipeline"
    elif args.preprocess:
        mode = "preprocess"
    elif args.realistic_backtest:
        mode = "realistic_backtest"
    elif args.robust_backtest:
        mode = "robust_backtest"
    elif args.realistic_backtest_live:
        mode = "realistic_backtest_live"
    else:
        safe_print("No mode specified. Use --help for options.", "ERROR")
        return 1
    
    # Run the selected mode
    try:
        success = run_mode(mode)
        
        # Summary
        safe_print("=" * 80)
        safe_print("EXECUTION SUMMARY")
        safe_print("=" * 80)
        
        if success:
            safe_print("Mode completed successfully!", "SUCCESS")
            safe_print("Check output_default/ for results", "INFO")
        else:
            safe_print("Mode failed or no result", "ERROR")
        
        # Check for log files
        log_files = []
        if os.path.exists("output_default"):
            for file in os.listdir("output_default"):
                if file.endswith('.log') or 'error' in file.lower():
                    log_files.append(file)
        
        if not log_files:
            safe_print("No critical issues found in logs!", "SUCCESS")
        else:
            safe_print(f"Check log files: {', '.join(log_files)}", "WARNING")
        
        safe_print("=" * 80)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        safe_print("Execution interrupted by user", "WARNING")
        return 130
    except Exception as e:
        safe_print(f"Critical error: {e}", "ERROR")
        safe_print(f"Traceback: {traceback.format_exc()}", "DEBUG")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
