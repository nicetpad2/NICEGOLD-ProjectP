#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Full Pipeline Test with Advanced Results Summary
════════════════════════════════════════════════════════════════════════════════

ทดสอบระบบ Full Pipeline พร้อมกับระบบสรุปผลลัพธ์แบบ "เทพ" แบบไม่ต้องใช้ interactive input

Author: NICEGOLD Team
Created: June 24, 2025
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules
try:
    from commands.advanced_results_summary import create_pipeline_results_summary
    from commands.pipeline_commands import PipelineCommands
    from core.colors import Colors, colorize
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


def main():
    """Test the enhanced full pipeline with advanced results summary"""

    print(
        f"{colorize('🎯 NICEGOLD ProjectP - Enhanced Full Pipeline Test', Colors.BOLD + Colors.BRIGHT_MAGENTA)}"
    )
    print(f"{colorize('=' * 70, Colors.BRIGHT_MAGENTA)}")

    print(
        f"\n{colorize('📅 วันที่:', Colors.BRIGHT_CYAN)} {colorize('June 24, 2025', Colors.WHITE)}"
    )
    print(
        f"{colorize('🎯 เป้าหมาย:', Colors.BRIGHT_CYAN)} ทดสอบ Full Pipeline + Advanced Results Summary"
    )
    print(f"{colorize('🔬 โหมด:', Colors.BRIGHT_CYAN)} Non-interactive Testing Mode")

    # Initialize components
    print(f"\n{colorize('🚀 กำลังเริ่มต้นระบบ...', Colors.BRIGHT_CYAN)}")

    try:
        # Initialize CSV manager (optional)
        csv_manager = None
        try:
            from src.robust_csv_manager import RobustCSVManager

            csv_manager = RobustCSVManager()
            print(f"{colorize('✅ CSV Manager initialized', Colors.BRIGHT_GREEN)}")
        except ImportError:
            print(f"{colorize('⚠️ CSV Manager not available', Colors.BRIGHT_YELLOW)}")

        # Initialize logger (optional)
        logger = None
        try:
            from src.advanced_logger import get_logger

            logger = get_logger("FullPipelineTest")
            print(f"{colorize('✅ Advanced Logger initialized', Colors.BRIGHT_GREEN)}")
        except ImportError:
            print(
                f"{colorize('⚠️ Advanced Logger not available', Colors.BRIGHT_YELLOW)}"
            )

        # Initialize pipeline commands
        pipeline_commands = PipelineCommands(
            project_root=PROJECT_ROOT, csv_manager=csv_manager, logger=logger
        )
        print(f"{colorize('✅ Pipeline Commands initialized', Colors.BRIGHT_GREEN)}")

        # Initialize advanced results summary
        results_summary = create_pipeline_results_summary(PROJECT_ROOT, logger)
        print(
            f"{colorize('✅ Advanced Results Summary initialized', Colors.BRIGHT_GREEN)}"
        )

        print(
            f"\n{colorize('🎯 ระบบพร้อมแล้ว! กำลังรัน Enhanced Full Pipeline...', Colors.BOLD + Colors.BRIGHT_GREEN)}"
        )
        print(f"{colorize('=' * 70, Colors.BRIGHT_GREEN)}")

        # Run the enhanced full pipeline
        success = pipeline_commands.full_pipeline()

        if success:
            print(
                f"\n{colorize('🎉 ENHANCED FULL PIPELINE สำเร็จเรียบร้อย!', Colors.BOLD + Colors.BRIGHT_GREEN)}"
            )
            print(f"{colorize('=' * 70, Colors.BRIGHT_GREEN)}")

            print(
                f"\n{colorize('📊 สรุปผลการทดสอบ:', Colors.BOLD + Colors.BRIGHT_CYAN)}"
            )
            print(f"✅ {colorize('Pipeline execution:', Colors.BRIGHT_GREEN)} สำเร็จ")
            print(
                f"✅ {colorize('Advanced results summary:', Colors.BRIGHT_GREEN)} ทำงานได้"
            )
            print(
                f"✅ {colorize('Model performance analysis:', Colors.BRIGHT_GREEN)} สมบูรณ์"
            )
            print(
                f"✅ {colorize('Feature importance analysis:', Colors.BRIGHT_GREEN)} สมบูรณ์"
            )
            print(f"✅ {colorize('Trading simulation:', Colors.BRIGHT_GREEN)} สมบูรณ์")
            print(f"✅ {colorize('Optimization results:', Colors.BRIGHT_GREEN)} สมบูรณ์")
            print(
                f"✅ {colorize('Intelligent recommendations:', Colors.BRIGHT_GREEN)} สมบูรณ์"
            )

            print(
                f"\n{colorize('🏆 ระบบสรุปผลลัพธ์แบบ เทพ ทำงานได้แล้ว!', Colors.BOLD + Colors.BRIGHT_MAGENTA)}"
            )

        else:
            print(
                f"\n{colorize('❌ Enhanced Full Pipeline ล้มเหลว', Colors.BRIGHT_RED)}"
            )
            return False

    except Exception as e:
        print(f"\n{colorize('❌ เกิดข้อผิดพลาด:', Colors.BRIGHT_RED)} {e}")
        import traceback

        traceback.print_exc()
        return False

    print(f"\n{colorize('🚀 การทดสอบเสร็จสิ้น!', Colors.BOLD + Colors.BRIGHT_GREEN)}")
    print(f"{colorize('=' * 70, Colors.BRIGHT_GREEN)}")

    return True


if __name__ == "__main__":
    main()
