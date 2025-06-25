#!/usr/bin/env python3
from datetime import datetime
                from ProjectP import run_full_pipeline, main
import argparse
import json
import os
            import psutil
import sys
import time
                    import traceback
"""
Production - ready ProjectP Mode Controller
ตัวควบคุมโหมดการทำงานของ ProjectP แบบ Production
"""


class ProjectPModeController:
    """ควบคุมโหมดการทำงานของ ProjectP"""

    def __init__(self):
        self.modes = {
            "default": {
                "description": "โหมดมาตรฐาน - ใช้ทุกฟีเจอร์", 
                "drift_monitor": True, 
                "schema_validation": True, 
                "resource_logging": True, 
                "performance_logging": True, 
                "export_summary": True, 
                "timeout": 1800  # 30 minutes
            }, 
            "debug": {
                "description": "โหมดดีบัก - ละเอียดสูงสุด", 
                "drift_monitor": True, 
                "schema_validation": True, 
                "resource_logging": True, 
                "performance_logging": True, 
                "export_summary": True, 
                "verbose_logging": True, 
                "detailed_errors": True, 
                "timeout": 3600  # 1 hour
            }, 
            "fast": {
                "description": "โหมดเร็ว - ข้ามการตรวจสอบส่วนใหญ่", 
                "drift_monitor": False, 
                "schema_validation": False, 
                "resource_logging": False, 
                "performance_logging": False, 
                "export_summary": False, 
                "timeout": 600  # 10 minutes
            }, 
            "ultimate": {
                "description": "โหมดสุดยอด - ทุกฟีเจอร์ + การตรวจสอบเพิ่มเติม", 
                "drift_monitor": True, 
                "schema_validation": True, 
                "resource_logging": True, 
                "performance_logging": True, 
                "export_summary": True, 
                "advanced_features": True, 
                "quality_checks": True, 
                "model_validation": True, 
                "timeout": 7200  # 2 hours
            }, 
            "production": {
                "description": "โหมดโปรดักชัน - เสถียรสูงสุด", 
                "drift_monitor": True, 
                "schema_validation": True, 
                "resource_logging": True, 
                "performance_logging": True, 
                "export_summary": True, 
                "error_handling": "strict", 
                "backup_enabled": True, 
                "safety_checks": True, 
                "timeout": 2400  # 40 minutes
            }
        }

        self.current_mode = "default"
        self.start_time = None

    def get_mode_config(self, mode_name):
        """ดึงการตั้งค่าของโหมดที่ระบุ"""
        return self.modes.get(mode_name, self.modes["default"])

    def list_modes(self):
        """แสดงรายการโหมดทั้งหมด"""
        print("📋 รายการโหมดที่รองรับ:")
        print(" = " * 50)

        for mode_name, config in self.modes.items():
            print(f"🔧 {mode_name.upper()}")
            print(f"   📝 {config['description']}")
            print(f"   ⏱️ Timeout: {config['timeout']} วินาที")

            features = []
            if config.get('drift_monitor'):
                features.append("Drift Monitor")
            if config.get('schema_validation'):
                features.append("Schema Validation")
            if config.get('advanced_features'):
                features.append("Advanced Features")
            if config.get('safety_checks'):
                features.append("Safety Checks")

            if features:
                print(f"   ✅ ฟีเจอร์: {', '.join(features)}")
            print()

    def validate_mode(self, mode_name):
        """ตรวจสอบว่าโหมดที่ระบุมีอยู่หรือไม่"""
        if mode_name not in self.modes:
            print(f"❌ โหมด '{mode_name}' ไม่มีอยู่")
            print("📋 โหมดที่รองรับ:", ", ".join(self.modes.keys()))
            return False
        return True

    def run_projectp_with_mode(self, mode_name, pipeline_args = None):
        """รัน ProjectP ด้วยโหมดที่ระบุ"""
        if not self.validate_mode(mode_name):
            return False

        self.current_mode = mode_name
        mode_config = self.get_mode_config(mode_name)
        self.start_time = time.time()

        print(f"🚀 กำลังรัน ProjectP ในโหมด {mode_name.upper()}")
        print(f"📝 {mode_config['description']}")
        print(f"⏱️ เริ่มต้น: {datetime.now()}")
        print(f"⏰ Timeout: {mode_config['timeout']} วินาที")
        print()

        try:
            # Import ProjectP modules
            sys.path.insert(0, '.')

            # Load fallbacks if needed
            if os.path.exists('pydantic_fallback.py'):
                exec(open('pydantic_fallback.py').read())
                print("✅ Pydantic fallback loaded")

            if os.path.exists('sklearn_fallback.py'):
                exec(open('sklearn_fallback.py').read())
                print("✅ Sklearn fallback loaded")

            # Set environment variables for mode
            os.environ['PROJECTP_MODE'] = mode_name
            os.environ['PROJECTP_TIMEOUT'] = str(mode_config['timeout'])

            if mode_config.get('verbose_logging'):
                os.environ['PROJECTP_VERBOSE'] = '1'

            if mode_config.get('error_handling') == 'strict':
                os.environ['PROJECTP_STRICT_MODE'] = '1'

            # Import and run ProjectP
            print("📦 Loading ProjectP...")

            # Try different import methods
            try:
                print("✅ ProjectP imported successfully")

                # Run based on pipeline args
                if pipeline_args and ' -  - run_full_pipeline' in pipeline_args:
                    print("🔄 Running full pipeline...")
                    result = run_full_pipeline()
                else:
                    print("🎯 Running main function...")
                    result = main()

                runtime = time.time() - self.start_time
                print(f"\n✅ ProjectP completed in {runtime:.1f} seconds")

                # Post - processing based on mode
                self.post_process_results(mode_config, runtime)

                return True

            except Exception as e:
                print(f"❌ ProjectP execution error: {e}")

                if mode_config.get('detailed_errors'):
                    traceback.print_exc()

                return False

        except KeyboardInterrupt:
            print("\n⏹️ หยุดการทำงานโดยผู้ใช้")
            return False
        except Exception as e:
            print(f"\n❌ ข้อผิดพลาด: {e}")
            return False

    def post_process_results(self, mode_config, runtime):
        """ประมวลผลหลังการรัน"""
        print("\n📊 Post - processing results...")

        # Check for output files
        result_files = [
            'classification_report.json', 
            'features_main.json', 
            'system_info.json', 
            'predictions.csv'
        ]

        results_summary = {
            'mode': self.current_mode, 
            'runtime_seconds': runtime, 
            'timestamp': datetime.now().isoformat(), 
            'files_created': [], 
            'performance': {}
        }

        for filename in result_files:
            if os.path.exists(filename):
                stat = os.stat(filename)
                size_mb = stat.st_size / (1024 * 1024)
                results_summary['files_created'].append({
                    'filename': filename, 
                    'size_mb': round(size_mb, 2), 
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
                print(f"   ✅ {filename}: {size_mb:.2f}MB")

        # Performance analysis
        if os.path.exists('classification_report.json'):
            try:
                with open('classification_report.json', 'r') as f:
                    data = json.load(f)

                accuracy = data.get('accuracy', 0)
                results_summary['performance']['accuracy'] = accuracy

                if accuracy >= 0.95:
                    performance_rating = "EXCELLENT"
                elif accuracy >= 0.85:
                    performance_rating = "GOOD"
                elif accuracy >= 0.70:
                    performance_rating = "FAIR"
                else:
                    performance_rating = "NEEDS_IMPROVEMENT"

                results_summary['performance']['rating'] = performance_rating

                print(f"   🎯 Accuracy: {accuracy:.4f} ({performance_rating})")

            except Exception as e:
                print(f"   ⚠️ Error reading performance data: {e}")

        # Save mode summary
        if mode_config.get('export_summary'):
            summary_file = f"mode_summary_{self.current_mode}_{int(time.time())}.json"

            with open(summary_file, 'w', encoding = 'utf - 8') as f:
                json.dump(results_summary, f, indent = 2, ensure_ascii = False)

            print(f"   📄 Summary saved to {summary_file}")

        # Mode - specific post - processing
        if mode_config.get('safety_checks'):
            self.run_safety_checks()

        if mode_config.get('quality_checks'):
            self.run_quality_checks()

    def run_safety_checks(self):
        """รันการตรวจสอบความปลอดภัย"""
        print("🛡️ Running safety checks...")

        checks = []

        # Check for critical files
        if os.path.exists('classification_report.json'):
            checks.append("✅ Classification report exists")
        else:
            checks.append("❌ Classification report missing")

        # Check system resources
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                checks.append("⚠️ High memory usage")
            else:
                checks.append("✅ Memory usage normal")
        except:
            checks.append("⚠️ Could not check memory")

        for check in checks:
            print(f"   {check}")

    def run_quality_checks(self):
        """รันการตรวจสอบคุณภาพ"""
        print("🔍 Running quality checks...")

        checks = []

        # Check model performance
        if os.path.exists('classification_report.json'):
            try:
                with open('classification_report.json', 'r') as f:
                    data = json.load(f)

                accuracy = data.get('accuracy', 0)

                if accuracy >= 0.95:
                    checks.append("✅ Excellent model performance")
                elif accuracy >= 0.80:
                    checks.append("✅ Good model performance")
                else:
                    checks.append("⚠️ Model performance needs improvement")

            except:
                checks.append("❌ Could not evaluate model performance")

        # Check data quality
        if os.path.exists('features_main.json'):
            checks.append("✅ Features file available")
        else:
            checks.append("❌ Features file missing")

        for check in checks:
            print(f"   {check}")

def main():
    """Main function for mode controller"""
    parser = argparse.ArgumentParser(description = 'ProjectP Mode Controller')
    parser.add_argument(' -  - mode', ' - m', default = 'default', 
                       help = 'Mode to run (default, debug, fast, ultimate, production)')
    parser.add_argument(' -  - list - modes', action = 'store_true', 
                       help = 'List all available modes')
    parser.add_argument(' -  - run_full_pipeline', action = 'store_true', 
                       help = 'Run full pipeline')

    args = parser.parse_args()

    controller = ProjectPModeController()

    if args.list_modes:
        controller.list_modes()
        return

    # Prepare pipeline args
    pipeline_args = []
    if args.run_full_pipeline:
        pipeline_args.append(' -  - run_full_pipeline')

    # Run ProjectP with specified mode
    success = controller.run_projectp_with_mode(args.mode, pipeline_args)

    if success:
        print("\n🎉 ProjectP completed successfully!")
    else:
        print("\n❌ ProjectP execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()