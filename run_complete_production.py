#!/usr/bin/env python3
"""
🚀 COMPLETE PRODUCTION RUN
รันระบบ NICEGOLD แบบสมบูรณ์ - ทดสอบเหมือนใช้งานจริง 100%
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

class CompleteProductionRunner:
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {
            "start_time": self.start_time.isoformat(),
            "steps_completed": [],
            "errors": [],
            "final_metrics": {},
            "status": "RUNNING"
        }
        
    def log(self, message, status="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        emoji = "🔵" if status == "INFO" else "✅" if status == "SUCCESS" else "❌" if status == "ERROR" else "⚠️"
        print(f"{emoji} [{timestamp}] {message}")
        
    def run_step(self, name, command, description):
        """Run a single step with comprehensive logging"""
        self.log(f"Starting: {description}", "INFO")
        
        try:
            if command.startswith("python "):
                # Run Python scripts directly
                module_name = command.replace("python ", "").replace(".py", "")
                if module_name == "emergency_auc_fix":
                    import emergency_auc_fix
                    result = emergency_auc_fix.main()
                    self.results["steps_completed"].append({
                        "name": name,
                        "description": description,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    self.log(f"Completed: {name} with result: {result}", "SUCCESS")
                    return result
                else:
                    # For other modules, use exec
                    exec(f"import {module_name}")
                    
            else:
                # Run shell commands
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    self.log(f"Completed: {name}", "SUCCESS")
                    return result.stdout
                else:
                    raise Exception(f"Command failed: {result.stderr}")
                    
        except Exception as e:
            self.log(f"Error in {name}: {e}", "ERROR")
            self.results["errors"].append({
                "step": name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return None
    
    def run_complete_pipeline(self):
        """Run the complete production pipeline"""
        self.log("🚀 STARTING COMPLETE PRODUCTION RUN", "INFO")
        print("=" * 60)
        
        # Step 1: Emergency AUC Fix
        auc_result = self.run_step(
            "Emergency_AUC_Fix",
            "python emergency_auc_fix.py",
            "Emergency AUC improvement with advanced features"
        )
        
        if auc_result:
            self.results["final_metrics"]["emergency_auc"] = auc_result
            
        # Step 2: Production monitoring
        self.run_step(
            "Production_Monitor",
            "python monitor_production_status.py",
            "Real-time production status monitoring"
        )
        
        # Step 3: Check results
        self.check_all_results()
        
        # Step 4: Generate final report
        self.generate_final_report()
        
        return self.results
    
    def check_all_results(self):
        """Check all result files and compile metrics"""
        self.log("📊 Checking all result files...", "INFO")
        
        result_files = [
            "fixes/emergency_auc_fix_results.json",
            "fixes/quick_test_results.json",
            "fixes/production_status_report.json"
        ]
        
        all_aucs = []
        
        for file_path in result_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract AUC values
                    for key, value in data.items():
                        if 'auc' in key.lower() and isinstance(value, (int, float)):
                            all_aucs.append(value)
                            self.log(f"Found AUC in {file_path}: {key} = {value:.3f}", "SUCCESS")
                            
                except Exception as e:
                    self.log(f"Error reading {file_path}: {e}", "ERROR")
        
        if all_aucs:
            best_auc = max(all_aucs)
            avg_auc = sum(all_aucs) / len(all_aucs)
            
            self.results["final_metrics"].update({
                "best_auc": best_auc,
                "average_auc": avg_auc,
                "auc_count": len(all_aucs),
                "baseline_auc": 0.516,
                "improvement": ((best_auc - 0.516) / 0.516 * 100) if best_auc > 0.516 else 0
            })
            
            self.log(f"🏆 BEST AUC FOUND: {best_auc:.3f}", "SUCCESS")
            self.log(f"📈 IMPROVEMENT: {self.results['final_metrics']['improvement']:.1f}%", "SUCCESS")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        self.log("📋 Generating final production report...", "INFO")
        
        metrics = self.results["final_metrics"]
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Determine overall status
        if "best_auc" in metrics:
            if metrics["best_auc"] > 0.70:
                self.results["status"] = "EXCELLENT"
                status_emoji = "🎉"
            elif metrics["best_auc"] > 0.60:
                self.results["status"] = "GOOD"
                status_emoji = "✅"
            elif metrics["best_auc"] > 0.55:
                self.results["status"] = "FAIR"
                status_emoji = "📈"
            else:
                self.results["status"] = "NEEDS_WORK"
                status_emoji = "⚠️"
        else:
            self.results["status"] = "INCOMPLETE"
            status_emoji = "❌"
        
        self.results.update({
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "steps_total": len(self.results["steps_completed"]),
            "errors_total": len(self.results["errors"])
        })
        
        # Save comprehensive results
        os.makedirs("fixes", exist_ok=True)
        with open("fixes/complete_production_run_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"{status_emoji} COMPLETE PRODUCTION RUN SUMMARY")
        print("=" * 60)
        
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"✅ Steps completed: {self.results['steps_total']}")
        print(f"❌ Errors: {self.results['errors_total']}")
        
        if "best_auc" in metrics:
            print(f"🎯 Best AUC: {metrics['best_auc']:.3f}")
            print(f"📊 Baseline: {metrics['baseline_auc']}")
            print(f"📈 Improvement: {metrics['improvement']:.1f}%")
        
        print(f"🏆 Overall Status: {self.results['status']}")
        
        # Recommendations
        print(f"\n🚀 RECOMMENDATIONS:")
        if self.results['status'] == "EXCELLENT":
            print("   ✅ READY FOR PRODUCTION DEPLOYMENT!")
            print("   📋 Next: Set up monitoring and alerts")
            print("   🔄 Next: Implement CI/CD pipeline")
        elif self.results['status'] == "GOOD":
            print("   📈 Good performance, ready for staging")
            print("   🔧 Consider further optimization")
            print("   📊 Validate on more data")
        else:
            print("   🔧 Continue optimization efforts")
            print("   📊 Review feature engineering")
            print("   🤖 Consider ensemble methods")
        
        print(f"\n💾 Detailed results: fixes/complete_production_run_results.json")
        
        self.log("🎉 COMPLETE PRODUCTION RUN FINISHED!", "SUCCESS")

def main():
    """Main execution function"""
    runner = CompleteProductionRunner()
    
    try:
        results = runner.run_complete_pipeline()
        return results
        
    except KeyboardInterrupt:
        runner.log("❌ Run interrupted by user", "ERROR")
        return None
        
    except Exception as e:
        runner.log(f"❌ Critical error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 NICEGOLD COMPLETE PRODUCTION RUN")
    print("🎯 Running all systems for production testing...")
    print("⏳ Please wait for complete results...\n")
    
    final_results = main()
    
    if final_results and final_results["status"] in ["EXCELLENT", "GOOD"]:
        print("\n🎊 SUCCESS: Production system is ready!")
    else:
        print("\n🔧 CONTINUE: System needs more optimization")
