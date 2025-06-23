#!/usr/bin/env python3
"""
ProjectP Production Mode Runner - รัน ProjectP ในทุกโหมดแบบ Production
รองรับ: Default, Debug, Fast, Ultimate, Production + Safe Mode
"""

import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import core modules
try:
    from core.config import config_manager
    from core.resource_monitor import resource_monitor
    from core.display import banner_manager, progress_display
    from core.pipeline_modes import pipeline_manager
    HAS_CORE_MODULES = True
except ImportError as e:
    print(f"❌ Error importing core modules: {e}")
    HAS_CORE_MODULES = False


class AllModesRunner:
    """Class สำหรับรันทุกโหมดและทดสอบ"""
    
    def __init__(self):
        if HAS_CORE_MODULES:
            self.config = config_manager
            self.monitor = resource_monitor
            self.banner = banner_manager
            self.progress = progress_display
            self.pipeline = pipeline_manager
        self.results = {}
        self.start_time = time.time()
    
    def run_all_modes(self) -> Dict[str, Any]:
        """รันทุกโหมดตามลำดับ"""
        
        if not HAS_CORE_MODULES:
            print("❌ Core modules not available. Cannot run all modes.")
            return {"error": "Core modules not available"}
        
        # แสดงแบนเนอร์
        self.banner.print_professional_banner()
        
        # ตรวจสอบทรัพยากรก่อนเริ่ม
        self.banner.print_section_header("PRE-EXECUTION RESOURCE CHECK", "🔍")
        self.monitor.print_resource_summary()
        
        # รายการโหมดที่จะทดสอบ
        modes_to_test = [
            ("preprocess", "⚙️ Preprocessing Mode"),
            ("class_balance_fix", "🎯 Class Balance Fix Mode"),
            ("realistic_backtest", "📈 Realistic Backtest Mode"),
            ("robust_backtest", "🛡️ Robust Backtest Mode"),
            ("realistic_backtest_live", "📊 Live Backtest Mode"),
            ("run_full_pipeline", "🚀 Full Pipeline Mode"),
            ("debug_full_pipeline", "🔍 Debug Pipeline Mode"),
            ("ultimate_pipeline", "🔥 Ultimate Pipeline Mode")
        ]
        
        self.banner.print_section_header("STARTING ALL MODES TEST", "🚀")
        print(f"📋 Total modes to test: {len(modes_to_test)}")
        print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # รันแต่ละโหมด
        for i, (mode_key, mode_name) in enumerate(modes_to_test, 1):
            print(f"\n{'='*60}")
            print(f"🔄 [{i}/{len(modes_to_test)}] Testing: {mode_name}")
            print(f"{'='*60}")
            
            # วัดเวลา
            mode_start_time = time.time()
            
            try:
                # เรียกใช้โหมด
                result = self.run_single_mode(mode_key)
                mode_duration = time.time() - mode_start_time
                
                # บันทึกผลลัพธ์
                self.results[mode_key] = {
                    "name": mode_name,
                    "status": "success" if result else "warning",
                    "result": result,
                    "duration": mode_duration,
                    "timestamp": datetime.now().isoformat()
                }
                
                if result:
                    self.banner.print_success(f"✅ {mode_name} completed successfully")
                    self.banner.print_info(f"📊 Result: {result}")
                else:
                    self.banner.print_warning(f"⚠️ {mode_name} completed with warnings")
                
                self.banner.print_info(f"⏱️ Duration: {mode_duration:.2f} seconds")
                
            except Exception as e:
                mode_duration = time.time() - mode_start_time
                self.results[mode_key] = {
                    "name": mode_name,
                    "status": "error",
                    "result": None,
                    "error": str(e),
                    "duration": mode_duration,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.banner.print_error(f"❌ {mode_name} failed: {e}")
                self.banner.print_info(f"⏱️ Duration: {mode_duration:.2f} seconds")
            
            # ตรวจสอบทรัพยากรหลังจากแต่ละโหมด
            if i < len(modes_to_test):  # ไม่ต้องตรวจสอบหลังโหมดสุดท้าย
                print(f"\n🧹 Optimizing memory before next mode...")
                self.monitor.optimize_memory()
                time.sleep(1)  # พักสั้นๆ
        
        # สรุปผลลัพธ์
        self.print_final_summary()
        
        return self.results
    
    def run_single_mode(self, mode_key: str) -> str:
        """รันโหมดเดียว"""
        
        mode_functions = {
            "preprocess": self.pipeline.run_preprocess_mode,
            "class_balance_fix": self.pipeline.run_class_balance_fix_mode,
            "realistic_backtest": self.pipeline.run_realistic_backtest_mode,
            "robust_backtest": self.pipeline.run_robust_backtest_mode,
            "realistic_backtest_live": self.pipeline.run_realistic_backtest_live_mode,
            "run_full_pipeline": self.pipeline.run_full_mode,
            "debug_full_pipeline": self.pipeline.run_debug_mode,
            "ultimate_pipeline": self.pipeline.run_ultimate_mode
        }
        
        if mode_key in mode_functions:
            return mode_functions[mode_key]()
        else:
            raise ValueError(f"Unknown mode: {mode_key}")
    
    def print_final_summary(self) -> None:
        """พิมพ์สรุปผลลัพธ์สุดท้าย"""
        
        total_duration = time.time() - self.start_time
        
        self.banner.print_section_header("🏁 FINAL TEST SUMMARY", "📊")
        
        # สถิติรวม
        total_modes = len(self.results)
        successful_modes = len([r for r in self.results.values() if r["status"] == "success"])
        warning_modes = len([r for r in self.results.values() if r["status"] == "warning"])
        failed_modes = len([r for r in self.results.values() if r["status"] == "error"])
        
        print(f"📋 Total modes tested: {total_modes}")
        self.banner.print_success(f"✅ Successful: {successful_modes}")
        if warning_modes > 0:
            self.banner.print_warning(f"⚠️ With warnings: {warning_modes}")
        if failed_modes > 0:
            self.banner.print_error(f"❌ Failed: {failed_modes}")
        
        print(f"⏱️ Total execution time: {total_duration:.2f} seconds")
        
        # รายละเอียดแต่ละโหมด
        self.banner.print_section_header("📋 DETAILED RESULTS", "📝")
        
        for mode_key, result in self.results.items():
            print(f"\n🔸 {result['name']}:")
            
            if result['status'] == 'success':
                self.banner.print_success(f"   Status: {result['status'].upper()}")
            elif result['status'] == 'warning':
                self.banner.print_warning(f"   Status: {result['status'].upper()}")
            else:
                self.banner.print_error(f"   Status: {result['status'].upper()}")
            
            print(f"   Duration: {result['duration']:.2f}s")
            
            if result.get('result'):
                print(f"   Result: {result['result']}")
            
            if result.get('error'):
                print(f"   Error: {result['error']}")
        
        # คำแนะนำ
        self.banner.print_section_header("💡 RECOMMENDATIONS", "🎯")
        
        if failed_modes == 0:
            self.banner.print_success("🎉 All modes completed successfully!")
            self.banner.print_info("✨ Your refactored ProjectP structure is working perfectly!")
        elif failed_modes < total_modes / 2:
            self.banner.print_warning("⚠️ Some modes failed, but majority are working")
            self.banner.print_info("🔧 Check failed modes and fix specific issues")
        else:
            self.banner.print_error("❌ Many modes failed")
            self.banner.print_info("🚨 Review the refactored structure and dependencies")
        
        # ข้อมูลทรัพยากรสุดท้าย
        self.banner.print_section_header("🖥️ FINAL RESOURCE STATUS", "📊")
        self.monitor.print_resource_summary()
        
        # บันทึกผลลัพธ์
        self.save_results()
    
    def save_results(self) -> None:
        """บันทึกผลลัพธ์ลงไฟล์"""
        try:
            import json
            
            # เพิ่มข้อมูลเมตา
            full_results = {
                "test_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_duration": time.time() - self.start_time,
                    "total_modes": len(self.results),
                    "python_version": sys.version,
                    "platform": sys.platform
                },
                "results": self.results
            }
            
            # บันทึกเป็น JSON
            output_file = f"all_modes_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            
            self.banner.print_success(f"📄 Results saved to: {output_file}")
            
        except Exception as e:
            self.banner.print_warning(f"⚠️ Could not save results: {e}")

    def quick_test(self) -> None:
        """ทดสอบแบบเร็ว (เฉพาะโหมดสำคัญ)"""
        
        if not HAS_CORE_MODULES:
            print("❌ Core modules not available. Cannot run quick test.")
            return
        
        self.banner.print_professional_banner()
        
        # ทดสอบเฉพาะโหมดหลัก
        quick_modes = [
            ("preprocess", "⚙️ Preprocessing Mode"),
            ("run_full_pipeline", "🚀 Full Pipeline Mode"),
            ("class_balance_fix", "🎯 Class Balance Fix Mode")
        ]
        
        self.banner.print_section_header("🚀 QUICK TEST MODE", "⚡")
        print(f"📋 Testing {len(quick_modes)} essential modes")
        
        for mode_key, mode_name in quick_modes:
            print(f"\n🔄 Testing: {mode_name}")
            
            try:
                start_time = time.time()
                result = self.run_single_mode(mode_key)
                duration = time.time() - start_time
                
                if result:
                    self.banner.print_success(f"✅ {mode_name} - OK ({duration:.1f}s)")
                else:
                    self.banner.print_warning(f"⚠️ {mode_name} - Warning ({duration:.1f}s)")
                    
            except Exception as e:
                self.banner.print_error(f"❌ {mode_name} - Failed: {e}")
        
        self.banner.print_success("🎉 Quick test completed!")


def main():
    """Main function"""
    
    runner = AllModesRunner()
    
    # ตรวจสอบ arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            runner.quick_test()
        elif sys.argv[1] == "--help":
            print(__doc__)
            print("\nUsage:")
            print("  python run_all_modes.py           # Run all modes")
            print("  python run_all_modes.py --quick   # Quick test (essential modes only)")
            print("  python run_all_modes.py --help    # Show this help")
        else:
            print("❌ Unknown argument. Use --help for usage information.")
    else:
        # รันทุกโหมด
        runner.run_all_modes()


if __name__ == "__main__":
    main()