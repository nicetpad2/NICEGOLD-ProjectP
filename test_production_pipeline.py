#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NICEGOLD ProjectP - Production Test Runner
Quick test script for Production Full Pipeline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Test Production Full Pipeline"""
    print("🚀 Testing Production Full Pipeline...")

    try:
        from production_full_pipeline import ProductionFullPipeline

        # Create pipeline with production config
        config = {
            "data": {"folder": "datacsv"},
            "production": {
                "capital": 100,
                "risk_per_trade": 0.1,
                "min_auc_threshold": 0.70,
            },
        }

        pipeline = ProductionFullPipeline(min_auc_requirement=0.70, initial_capital=100)

        print("✅ Pipeline initialized successfully")

        # Run the pipeline
        print("🎯 Running Full Pipeline...")
        results = pipeline.run_full_pipeline()

        if results and results.get("success", False):
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")

            # Display results
            auc = results.get("auc", 0)
            model_name = results.get("model_name", "Unknown")
            backtest = results.get("backtest_results", {})

            print(f"\n📊 RESULTS:")
            print(f"   🎯 AUC Score: {auc:.3f}")
            print(f"   🤖 Best Model: {model_name}")

            if backtest:
                total_return = backtest.get("total_return", 0) * 100
                final_capital = backtest.get("final_capital", 100)
                sharpe_ratio = backtest.get("sharpe_ratio", 0)
                win_rate = backtest.get("win_rate", 0) * 100
                total_trades = backtest.get("total_trades", 0)

                print(f"   💰 Total Return: {total_return:+.2f}%")
                print(f"   💵 Final Capital: ${final_capital:.2f}")
                print(f"   📊 Sharpe Ratio: {sharpe_ratio:.2f}")
                print(f"   🎯 Win Rate: {win_rate:.1f}%")
                print(f"   📈 Total Trades: {total_trades}")

            print(
                f"\n✅ Production requirements: {'MET' if auc >= 0.70 else 'NOT MET'}"
            )

        else:
            print("❌ Pipeline failed to complete")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
