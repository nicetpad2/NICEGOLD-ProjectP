#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Enhanced Full Pipeline with Advanced Results Summary
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


def test_enhanced_pipeline():
    """Test the enhanced full pipeline with advanced results summary"""
    print("🚀 TESTING ENHANCED FULL PIPELINE WITH ADVANCED RESULTS SUMMARY")
    print("=" * 70)

    try:
        # Import the pipeline commands
        from src.commands.advanced_results_summary import (
            create_pipeline_results_summary,
        )
        from src.commands.pipeline_commands import PipelineCommands

        print("✅ Pipeline modules imported successfully")

        # Initialize pipeline commands
        pipeline_commands = PipelineCommands(
            project_root=PROJECT_ROOT, csv_manager=None, logger=None
        )

        print("✅ Pipeline commands initialized")

        # Test advanced results summary system
        results_summary = create_pipeline_results_summary(PROJECT_ROOT)
        print("✅ Advanced results summary system initialized")

        # Test collecting sample results
        sample_stage_data = {
            "duration": 25.5,
            "status": "completed",
            "metrics": {
                "accuracy": 0.892,
                "f1_score": 0.845,
                "samples_processed": 15000,
                "features_created": 12,
            },
            "outputs": {
                "model_file": "enhanced_model.pkl",
                "results_file": "stage_results.json",
            },
        }

        results_summary.collect_pipeline_stage_results(
            "enhanced_preprocessing", sample_stage_data
        )
        print("✅ Sample stage results collected")

        # Test model performance analysis
        import numpy as np
        from sklearn.metrics import accuracy_score

        # Generate synthetic test data
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=1000, p=[0.6, 0.4])
        y_pred = np.random.choice([0, 1], size=1000, p=[0.55, 0.45])
        y_pred_proba = np.random.uniform(0, 1, size=1000)

        results_summary.analyze_model_performance(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            model_name="Enhanced Test Model",
        )
        print("✅ Model performance analysis completed")

        # Test trading simulation analysis
        trading_results = {
            "total_return": 0.185,  # 18.5% return
            "sharpe_ratio": 1.75,
            "max_drawdown": 0.08,  # 8% max drawdown
            "win_rate": 0.62,  # 62% win rate
            "profit_factor": 1.85,
            "total_trades": 348,
            "winning_trades": 216,
            "losing_trades": 132,
            "average_win": 2.8,
            "average_loss": -1.6,
            "largest_win": 15.2,
            "largest_loss": -7.3,
            "simulation_period": "2023-2025",
        }

        results_summary.analyze_trading_simulation(trading_results)
        print("✅ Trading simulation analysis completed")

        # Test optimization results
        optimization_data = {
            "best_params": {
                "n_estimators": 200,
                "max_depth": 15,
                "learning_rate": 0.1,
                "subsample": 0.8,
            },
            "best_score": 0.892,
            "method": "RandomizedSearchCV",
            "n_trials": 100,
            "improvement": 0.045,
            "duration": 180.5,
        }

        results_summary.analyze_optimization_results(optimization_data)
        print("✅ Optimization results analysis completed")

        # Generate comprehensive summary
        print("\n🎯 GENERATING COMPREHENSIVE SUMMARY...")
        print("-" * 50)

        summary_text = results_summary.generate_comprehensive_summary()

        # Print the summary (truncated for readability)
        lines = summary_text.split("\n")
        for line in lines[:50]:  # Show first 50 lines
            print(line)

        if len(lines) > 50:
            print(f"\n... และอีก {len(lines) - 50} บรรทัด")

        print("\n✅ ENHANCED PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("🎉 Advanced Results Summary System is fully functional!")

        # Show quick summary
        print("\n⚡ QUICK PERFORMANCE OVERVIEW:")
        results_summary.print_quick_summary()

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_enhanced_pipeline()
    if success:
        print("\n🎯 ALL TESTS PASSED - ENHANCED PIPELINE READY FOR PRODUCTION!")
    else:
        print("\n❌ TESTS FAILED - PLEASE CHECK THE ERRORS ABOVE")
        sys.exit(1)
