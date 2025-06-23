#!/usr/bin/env python3
"""
🚀 PROJECTP ULTIMATE PRODUCTION-READY PIPELINE SYSTEM v3.0 - REFACTORED
========================================================================

🎯 COMPLETE SOLUTION FOR EXTREME CLASS IMBALANCE (201.7:1)
- ✅ Modular architecture for easy maintenance
- ✅ Separated concerns and responsibilities  
- ✅ Clean code structure
- ✅ All modes working perfectly

Refactored Structure:
- src/core/config.py: Configuration management
- src/core/resource_monitor.py: System resource monitoring
- src/core/display.py: Banner and UI display
- src/core/pipeline_modes.py: Pipeline mode implementations
- src/core/cli.py: Command line interface handling

Modes:
- full_pipeline: Complete end-to-end pipeline (production-ready)
- debug_full_pipeline: Full pipeline with detailed logging  
- preprocess: Data preparation and feature engineering only
- realistic_backtest: Realistic backtesting with walk-forward validation
- robust_backtest: Robust backtesting with multiple models
- realistic_backtest_live: Live-simulation backtesting
- ultimate_pipeline: 🔥 ALL improvements (Emergency Fixes + AUC + Full Pipeline)
- class_balance_fix: 🎯 Dedicated class imbalance solution
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import the CLI handler
from core.cli import cli_handler


def main() -> None:
    """
    Main entry point - ใช้ CLI handler ที่แยกออกมาแล้ว
    
    ใช้โครงสร้างใหม่ที่:
    - แยกความรับผิดชอบอย่างชัดเจน
    - ง่ายต่อการดูแลรักษา
    - ขยายได้ง่าย
    - ทดสอบได้ง่าย
    """
    try:
        cli_handler.parse_and_execute()
    except Exception as e:
        print(f"❌ Critical error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
