#!/usr/bin/env python3
from enhanced_visual_display import EnhancedReportGenerator, ThaiVisualDisplay
import time
"""
🎨 Demo Thai Visual Display System
แสดงระบบการแสดงผลภาษาไทยที่สวยงาม
"""


def demo_thai_display():
    """Demo the beautiful Thai visual display system"""

    # Initialize display
    display = ThaiVisualDisplay()
    report_generator = EnhancedReportGenerator()

    # Show welcome screen
    display.show_welcome_screen("🏆 NICEGOLD ProjectP - ระบบเทรดทองคำอัจฉริยะ")
    time.sleep(3)

    # Show system status
    print("\n📊 แสดงสถานะระบบ...")
    display.show_system_status(
        cpu_percent = 45.2, ram_percent = 67.8, gpu_available = True, gpu_usage = 23.5
    )
    time.sleep(2)

    # Demo progress with Thai stages
    print("\n🚀 แสดงความคืบหน้าของไปป์ไลน์...")

    thai_stages = [
        "🏗️ ขั้นตอนประมวลผลข้อมูล", 
        "🧠 สร้างฟีเจอร์ขั้นสูง", 
        "🤖 ฝึกสอนโมเดล", 
        "🎯 ปรับแต่งเกณฑ์", 
        "📊 สร้างรายงาน", 
    ]

    progress = display.create_progress_tracker()

    with progress:
        main_task = progress.add_task(
            "[bold gold1]🚀 ไปป์ไลน์ NICEGOLD ฉบับสมบูรณ์", total = len(thai_stages)
        )

        for i, stage_name in enumerate(thai_stages):
            stage_task = progress.add_task(f"[cyan]⏳ {stage_name}", total = 1)

            # Simulate processing
            for j in range(10):
                time.sleep(0.1)

            # Show stage summary
            display.show_stage_summary(
                {
                    stage_name: {
                        "success": True, 
                        "duration": 1.0 + i * 0.5, 
                        "details": f"ความแม่นยำ: {85 + i * 2}%", 
                    }
                }
            )

            progress.update(stage_task, completed = 1)
            progress.update(main_task, advance = 1)
            time.sleep(0.5)

    # Show final report
    print("\n🎉 แสดงรายงานสุดท้าย...")
    display.show_final_results(
        {
            "total_time": 15.7, 
            "successful_stages": 5, 
            "total_stages": 5, 
            "errors": [], 
            "accuracy": "94.5%", 
            "performance": "ดีเยี่ยม", 
        }
    )

    # Generate demo HTML report
    print("\n📈 สร้างรายงาน HTML...")
    demo_reports = {
        "performance": report_generator.generate_performance_report(
            {
                "total_time": 15.7, 
                "successful_stages": 5, 
                "total_stages": 5, 
                "stage_times": {
                    stage: 1.0 + i * 0.5 for i, stage in enumerate(thai_stages)
                }, 
                "peak_cpu": 78.3, 
                "peak_ram": 82.1, 
                "errors": [], 
                "warnings": [], 
            }
        ), 
        "data_quality": report_generator.generate_data_quality_report(
            {
                "data_validation": "passed", 
                "missing_values": 0, 
                "data_integrity": "high", 
                "total_records": 50000, 
                "valid_records": 49980, 
            }
        ), 
    }

    try:
        dashboard_path = report_generator.generate_html_dashboard(demo_reports)
        print("\n✅ Demo เสร็จสิ้น!")
        print(f"📊 รายงาน HTML: {dashboard_path}")
        print("💡 เปิดไฟล์ในเว็บเบราว์เซอร์เพื่อดูรายงานแบบโต้ตอบ")
    except Exception as e:
        print("\n✅ Demo เสร็จสิ้น!")
        print(f"📊 รายงาน HTML: ข้อผิดพลาด - {e}")
        print("💡 ระบบแสดงผลภาษาไทยทำงานได้อย่างสมบูรณ์!")


if __name__ == "__main__":
    demo_thai_display()