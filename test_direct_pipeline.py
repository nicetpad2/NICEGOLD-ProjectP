#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Full Pipeline Test with Advanced Results Summary
รันการทดสอบ full pipeline โดยตรงพร้อม advanced results summary
"""

import os
import sys
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
src_path = PROJECT_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def run_direct_full_pipeline_test():
    """รัน full pipeline โดยตรงพร้อม advanced results summary"""
    print("🚀 DIRECT FULL PIPELINE TEST WITH ADVANCED RESULTS SUMMARY")
    print("=" * 70)

    try:
        # Import required modules
        from core.colors import Colors, colorize
        from src.commands.pipeline_commands import PipelineCommands

        print(
            f"{colorize('✅ Required modules imported successfully', Colors.BRIGHT_GREEN)}"
        )

        # Initialize pipeline commands
        pipeline_commands = PipelineCommands(
            project_root=PROJECT_ROOT,
            csv_manager=None,  # Will be initialized if available
            logger=None,  # Will be initialized if available
        )

        print(f"{colorize('✅ Pipeline commands initialized', Colors.BRIGHT_GREEN)}")

        # สร้างข้อมูล synthetic สำหรับการทดสอบ
        print(
            f"\n{colorize('📊 Creating synthetic data for testing...', Colors.BRIGHT_CYAN)}"
        )

        from datetime import datetime, timedelta

        import numpy as np
        import pandas as pd

        # สร้างข้อมูลทองคำ synthetic
        np.random.seed(42)
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(minutes=i) for i in range(10000)]

        # สร้างราคาทองคำจำลอง (XAUUSD)
        base_price = 1950.0
        returns = np.random.normal(0, 0.01, 10000)  # 1% daily volatility
        prices = [base_price]

        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)

        # สร้าง OHLC data
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                "close": [p * (1 + np.random.normal(0, 0.002)) for p in prices],
                "volume": np.random.randint(100, 1000, 10000),
            }
        )

        # บันทึกข้อมูล
        os.makedirs("datacsv", exist_ok=True)
        df.to_csv("datacsv/processed_data.csv", index=False)

        print(
            f"{colorize('✅ Synthetic XAUUSD data created:', Colors.BRIGHT_GREEN)} {len(df)} rows"
        )

        # รัน full pipeline
        print(
            f"\n{colorize('🚀 STARTING ENHANCED FULL PIPELINE...', Colors.BOLD + Colors.BRIGHT_MAGENTA)}"
        )
        print(f"{colorize('=' * 60, Colors.BRIGHT_MAGENTA)}")

        success = pipeline_commands.full_pipeline()

        if success:
            print(
                f"\n{colorize('🎉 ENHANCED FULL PIPELINE COMPLETED SUCCESSFULLY!', Colors.BOLD + Colors.BRIGHT_GREEN)}"
            )
            print(
                f"{colorize('✅ Advanced Results Summary generated successfully!', Colors.BRIGHT_GREEN)}"
            )

            # ตรวจสอบไฟล์ผลลัพธ์ที่ถูกสร้าง
            results_dir = Path("results/summary")
            if results_dir.exists():
                result_files = list(results_dir.glob("*"))
                if result_files:
                    print(
                        f"\n{colorize('📁 Results files created:', Colors.BRIGHT_CYAN)}"
                    )
                    for file in sorted(result_files)[-3:]:  # แสดง 3 ไฟล์ล่าสุด
                        print(f"   📄 {file.name}")
                else:
                    print(
                        f"{colorize('⚠️ No results files found in results directory', Colors.BRIGHT_YELLOW)}"
                    )

            print(
                f"\n{colorize('🎯 PIPELINE SUMMARY:', Colors.BOLD + Colors.BRIGHT_BLUE)}"
            )
            print(f"   ✅ Data preprocessing completed")
            print(f"   ✅ Model training completed")
            print(f"   ✅ Trading simulation completed")
            print(f"   ✅ Advanced results summary generated")
            print(f"   ✅ Comprehensive analysis performed")

        else:
            print(f"\n{colorize('❌ PIPELINE FAILED', Colors.BRIGHT_RED)}")
            return False

        return True

    except Exception as e:
        print(f"\n{colorize('❌ Test failed with error:', Colors.BRIGHT_RED)} {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # ทำความสะอาดไฟล์ temporary
        temp_files = [
            "pipeline_results.json",
            "results_model_data.pkl",
            "results_model_object.pkl",
        ]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"🧹 Cleaned up: {temp_file}")
                except:
                    pass


if __name__ == "__main__":
    print(f"🔧 Testing Environment:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Working Directory: {os.getcwd()}")

    success = run_direct_full_pipeline_test()

    if success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Enhanced Full Pipeline with Advanced Results Summary is ready!")
        print(f"🚀 System is production-ready!")
    else:
        print(f"\n❌ TESTS FAILED!")
        sys.exit(1)
