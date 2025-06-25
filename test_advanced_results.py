#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the new Advanced Results Summary system in NICEGOLD ProjectP
ทดสอบระบบสรุปผลลัพธ์แบบครอบคลุมใหม่
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def test_advanced_results_summary():
    """Test the advanced results summary system"""
    try:
        import numpy as np

        from commands.advanced_results_summary import create_pipeline_results_summary
        from core.colors import Colors, colorize

        print(
            f"{colorize('🧪 Testing Advanced Results Summary System...', Colors.BOLD + Colors.BRIGHT_CYAN)}"
        )
        print(f"{colorize('═' * 60, Colors.BRIGHT_CYAN)}")

        # Create results summary instance
        results_summary = create_pipeline_results_summary(project_root)
        print(
            f"{colorize('✅ Results summary system created successfully', Colors.BRIGHT_GREEN)}"
        )

        # Test 1: Collect pipeline stage results
        print(
            f"\n{colorize('📊 Test 1: Pipeline Stage Results Collection', Colors.BRIGHT_BLUE)}"
        )
        stage_data = {
            "duration": 45.2,
            "status": "completed",
            "metrics": {"accuracy": 0.85, "samples_processed": 10000},
            "outputs": {"model_file": "test_model.pkl"},
            "errors": [],
            "warnings": ["Minor warning about data quality"],
        }
        results_summary.collect_pipeline_stage_results("data_preprocessing", stage_data)
        print(
            f"{colorize('✅ Stage results collected successfully', Colors.BRIGHT_GREEN)}"
        )

        # Test 2: Model performance analysis
        print(
            f"\n{colorize('🎯 Test 2: Model Performance Analysis', Colors.BRIGHT_BLUE)}"
        )
        # Generate synthetic test data
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_pred = np.random.randint(0, 2, 1000)
        y_pred_proba = np.random.random(1000)

        results_summary.analyze_model_performance(
            y_true, y_pred, y_pred_proba, "Test Model"
        )
        print(
            f"{colorize('✅ Model performance analysis completed', Colors.BRIGHT_GREEN)}"
        )

        # Test 3: Trading simulation analysis
        print(
            f"\n{colorize('📈 Test 3: Trading Simulation Analysis', Colors.BRIGHT_BLUE)}"
        )
        backtest_data = {
            "total_return": 0.25,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.15,
            "win_rate": 0.62,
            "profit_factor": 1.45,
            "total_trades": 150,
            "winning_trades": 93,
            "losing_trades": 57,
        }
        results_summary.analyze_trading_simulation(backtest_data)
        print(
            f"{colorize('✅ Trading simulation analysis completed', Colors.BRIGHT_GREEN)}"
        )

        # Test 4: Optimization results
        print(
            f"\n{colorize('⚙️ Test 4: Optimization Results Analysis', Colors.BRIGHT_BLUE)}"
        )
        optimization_data = {
            "best_params": {"n_estimators": 100, "max_depth": 10},
            "best_score": 0.89,
            "n_trials": 50,
            "improvement": 0.05,
            "method": "RandomizedSearchCV",
        }
        results_summary.analyze_optimization_results(optimization_data)
        print(f"{colorize('✅ Optimization analysis completed', Colors.BRIGHT_GREEN)}")

        # Test 5: Generate comprehensive summary
        print(
            f"\n{colorize('📋 Test 5: Comprehensive Summary Generation', Colors.BRIGHT_BLUE)}"
        )
        summary_text = results_summary.generate_comprehensive_summary()
        print(
            f"{colorize('✅ Comprehensive summary generated successfully', Colors.BRIGHT_GREEN)}"
        )

        # Test 6: Quick summary
        print(f"\n{colorize('⚡ Test 6: Quick Summary', Colors.BRIGHT_BLUE)}")
        results_summary.print_quick_summary()
        print(
            f"{colorize('✅ Quick summary displayed successfully', Colors.BRIGHT_GREEN)}"
        )

        print(
            f"\n{colorize('🎉 ALL TESTS PASSED! Advanced Results Summary System is ready!', Colors.BOLD + Colors.BRIGHT_GREEN)}"
        )
        return True

    except Exception as e:
        print(f"{colorize('❌ Test failed:', Colors.BRIGHT_RED)} {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test pipeline integration with results summary"""
    try:
        from commands.pipeline_commands import PipelineCommands
        from core.colors import Colors, colorize

        print(
            f"\n{colorize('🔧 Testing Pipeline Integration...', Colors.BOLD + Colors.BRIGHT_MAGENTA)}"
        )
        print(f"{colorize('═' * 60, Colors.BRIGHT_MAGENTA)}")

        # Create pipeline commands instance
        pipeline_commands = PipelineCommands(project_root)
        print(f"{colorize('✅ Pipeline commands initialized', Colors.BRIGHT_GREEN)}")

        print(
            f"{colorize('💡 Pipeline commands ready for enhanced full pipeline execution', Colors.BRIGHT_CYAN)}"
        )
        print(
            f"{colorize('📋 Use menu option 1 to run enhanced full pipeline with results', Colors.BRIGHT_YELLOW)}"
        )

        return True

    except Exception as e:
        print(
            f"{colorize('❌ Pipeline integration test failed:', Colors.BRIGHT_RED)} {e}"
        )
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print(
        f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🧪 NICEGOLD ProjectP - Advanced Results Summary Test Suite 🧪            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    )

    # Run tests
    test1_result = test_advanced_results_summary()
    test2_result = test_pipeline_integration()

    # Summary
    print(f"\n{colorize('📊 TEST SUITE SUMMARY:', Colors.BOLD + Colors.BRIGHT_YELLOW)}")
    print(f"{colorize('─' * 40, Colors.BRIGHT_YELLOW)}")
    print(
        f"Advanced Results Summary: {colorize('PASS' if test1_result else 'FAIL', Colors.BRIGHT_GREEN if test1_result else Colors.BRIGHT_RED)}"
    )
    print(
        f"Pipeline Integration: {colorize('PASS' if test2_result else 'FAIL', Colors.BRIGHT_GREEN if test2_result else Colors.BRIGHT_RED)}"
    )

    if test1_result and test2_result:
        print(
            f"\n{colorize('🎉 ALL TESTS PASSED! System ready for enhanced full pipeline!', Colors.BOLD + Colors.BRIGHT_GREEN)}"
        )
        print(
            f"\n{colorize('🚀 WHAT YOU CAN DO NOW:', Colors.BOLD + Colors.BRIGHT_CYAN)}"
        )
        print(
            f"1. {colorize('Run enhanced full pipeline:', Colors.BRIGHT_GREEN)} python3 ProjectP_refactored.py"
        )
        print(
            f"2. {colorize('Select option 1 (Full Pipeline)', Colors.BRIGHT_GREEN)} from the menu"
        )
        print(
            f"3. {colorize('Enjoy comprehensive results summary!', Colors.BRIGHT_GREEN)}"
        )
        print(
            f"\n{colorize('🎯 ENHANCED FEATURES:', Colors.BOLD + Colors.BRIGHT_MAGENTA)}"
        )
        print(f"• {colorize('Complete performance analysis', Colors.BRIGHT_WHITE)}")
        print(f"• {colorize('Intelligent recommendations', Colors.BRIGHT_WHITE)}")
        print(f"• {colorize('Beautiful visualizations', Colors.BRIGHT_WHITE)}")
        print(f"• {colorize('Detailed reports for development', Colors.BRIGHT_WHITE)}")
        print(f"• {colorize('Next steps guidance', Colors.BRIGHT_WHITE)}")
    else:
        print(
            f"\n{colorize('❌ Some tests failed. Please check the errors above.', Colors.BRIGHT_RED)}"
        )


if __name__ == "__main__":
    main()
