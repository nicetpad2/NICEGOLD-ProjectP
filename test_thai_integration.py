#!/usr/bin/env python3
        from enhanced_full_pipeline import EnhancedFullPipeline
from enhanced_visual_display import EnhancedReportGenerator, ThaiVisualDisplay
import sys
import traceback
"""
Test script to validate the Thai visual display integration with the main pipeline
"""


def test_visual_display_methods():
    """Test all methods of ThaiVisualDisplay"""
    print("🧪 Testing ThaiVisualDisplay methods...")

    try:
        # Initialize the display
        display = ThaiVisualDisplay()
        print("✅ ThaiVisualDisplay initialized")

        # Test progress tracker creation
        progress = display.create_progress_tracker()
        print("✅ create_progress_tracker() works")

        # Test system status
        display.show_system_status()
        print("✅ show_system_status() works")

        # Test stage summary
        display.show_stage_summary(
            {"test_stage": {"success": True, "duration": 2.5, "details": "Test passed"}}
        )
        print("✅ show_stage_summary() works")

        # Test final results
        display.show_final_results(
            {
                "total_time": 15.7, 
                "successful_stages": 5, 
                "total_stages": 5, 
                "accuracy": "95.2%", 
                "status": "เสร็จสมบูรณ์", 
            }
        )
        print("✅ show_final_results() works")

        print("🎉 All ThaiVisualDisplay methods work correctly!")
        return True

    except Exception as e:
        print(f"❌ Error testing ThaiVisualDisplay: {e}")
        traceback.print_exc()
        return False


def test_html_generator():
    """Test HTML report generator"""
    print("\n🧪 Testing EnhancedReportGenerator...")

    try:
        generator = EnhancedReportGenerator()
        print("✅ EnhancedReportGenerator initialized")

        # Test performance report
        perf_report = generator.generate_performance_report(
            {
                "total_time": 15.7, 
                "successful_stages": 5, 
                "total_stages": 5, 
                "stage_times": {"test": 2.5}, 
                "peak_cpu": 75.0, 
                "peak_ram": 65.0, 
                "errors": [], 
                "warnings": [], 
            }
        )
        print("✅ Performance report generated")

        # Test data quality report
        quality_report = generator.generate_data_quality_report(
            {
                "data_validation": "passed", 
                "missing_values": 0, 
                "data_integrity": "high", 
                "total_records": 1000, 
                "valid_records": 995, 
            }
        )
        print("✅ Data quality report generated")

        # Test HTML dashboard
        dashboard_path = generator.generate_html_dashboard(
            {"performance": perf_report, "data_quality": quality_report}
        )
        print(f"✅ HTML dashboard generated: {dashboard_path}")

        print("🎉 All EnhancedReportGenerator methods work correctly!")
        return True

    except Exception as e:
        print(f"❌ Error testing EnhancedReportGenerator: {e}")
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test pipeline integration"""
    print("\n🧪 Testing pipeline integration...")

    try:

        # Initialize pipeline (without running it)
        pipeline = EnhancedFullPipeline()
        print("✅ Pipeline initialized with Thai visual display")

        # Check if visual display is properly integrated
        if hasattr(pipeline, "visual_display"):
            print("✅ Pipeline has visual_display attribute")

            # Test if the visual display methods are accessible
            if hasattr(pipeline.visual_display, "create_progress_tracker"):
                print("✅ create_progress_tracker method is available")
            else:
                print("❌ create_progress_tracker method missing")
                return False

            if hasattr(pipeline.visual_display, "show_final_results"):
                print("✅ show_final_results method is available")
            else:
                print("❌ show_final_results method missing")
                return False
        else:
            print("❌ Pipeline missing visual_display attribute")
            return False

        print("🎉 Pipeline integration test passed!")
        return True

    except Exception as e:
        print(f"❌ Error testing pipeline integration: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Thai Visual Display Integration Tests\n")

    success_count = 0
    total_tests = 3

    # Test 1: Visual Display Methods
    if test_visual_display_methods():
        success_count += 1

    # Test 2: HTML Generator
    if test_html_generator():
        success_count += 1

    # Test 3: Pipeline Integration
    if test_pipeline_integration():
        success_count += 1

    # Final results
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print(
            "🎉 All integration tests passed! The Thai visual display system is ready for production."
        )
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())