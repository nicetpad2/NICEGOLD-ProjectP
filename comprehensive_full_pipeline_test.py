#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 COMPREHENSIVE FULL PIPELINE TEST
ทดสอบ Full Pipeline ทั้งระบบอย่างละเอียด
"""

import gc
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced logger
try:
    from enhanced_logger import EnhancedLogger
    logger = EnhancedLogger()
    info = logger.info
    success = logger.success
    warning = logger.warning
    error = logger.error
except ImportError:
    def info(msg): print(f"ℹ️  {msg}")
    def success(msg): print(f"✅ {msg}")
    def warning(msg): print(f"⚠️  {msg}")
    def error(msg): print(f"❌ {msg}")

# Resource monitoring


class ResourceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.measurements = []
    
    def measure(self, stage: str):
        current_time = time.time()
        current_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
        measurement = {
            'stage': stage,
            'time': current_time - self.start_time,
            'memory_mb': current_memory,
            'memory_diff_mb': current_memory - self.start_memory,
            'cpu_percent': psutil.cpu_percent(interval=0.1)
        }
        self.measurements.append(measurement)
        
        info(f"📊 {stage}")
        info(f"   ⏱️  Time: {measurement['time']:.1f}s")
        info(f"   🧠 Memory: {current_memory:.1f}MB "
             f"({measurement['memory_diff_mb']:+.1f}MB)")
        info(f"   💻 CPU: {measurement['cpu_percent']:.1f}%")
        
        return measurement
    
    def get_summary(self) -> Dict[str, Any]:
        total_time = (self.measurements[-1]['time']
                      if self.measurements else 0)
        max_memory = (max([m['memory_mb'] for m in self.measurements])
                      if self.measurements else 0)
        avg_cpu = (sum([m['cpu_percent'] for m in self.measurements]) /
                   len(self.measurements) if self.measurements else 0)
        
        return {
            'total_time': total_time,
            'start_memory_mb': self.start_memory,
            'peak_memory_mb': max_memory,
            'memory_increase_mb': max_memory - self.start_memory,
            'avg_cpu_percent': avg_cpu,
            'measurements': self.measurements
        }


class FullPipelineValidator:
    """ตรวจสอบ Full Pipeline ว่าทำงานครบลูปหรือไม่"""
    
    def __init__(self):
        self.monitor = ResourceMonitor()
        self.test_results = {}
        self.errors = []
        self.warnings = []
        
    def test_comprehensive_progress_system(self) -> Dict[str, Any]:
        """ทดสอบ ComprehensiveProgressSystem"""
        try:
            info("🔍 Testing ComprehensiveProgressSystem...")
            self.monitor.measure("Starting ComprehensiveProgressSystem test")
            
            from comprehensive_full_pipeline_progress import ComprehensiveProgressSystem
            progress_system = ComprehensiveProgressSystem()
            
            # Run pipeline
            results = progress_system.run_full_pipeline_with_complete_progress()
            
            self.monitor.measure("ComprehensiveProgressSystem completed")
            
            # Validate results
            if not isinstance(results, dict):
                self.errors.append("ComprehensiveProgressSystem did not return dict")
                return {"status": "FAILED", "error": "Invalid return type"}
            
            if results.get("success", False):
                success("✅ ComprehensiveProgressSystem test PASSED")
                return {"status": "SUCCESS", "results": results}
            else:
                warning("⚠️ ComprehensiveProgressSystem completed with issues")
                return {"status": "PARTIAL", "results": results}
                
        except Exception as e:
            error(f"❌ ComprehensiveProgressSystem test FAILED: {str(e)}")
            self.errors.append(f"ComprehensiveProgressSystem error: {str(e)}")
            return {"status": "FAILED", "error": str(e)}
    
    def test_production_pipeline(self) -> Dict[str, Any]:
        """ทดสอบ ProductionFullPipeline"""
        try:
            info("🔍 Testing ProductionFullPipeline...")
            self.monitor.measure("Starting ProductionFullPipeline test")
            
            from production_full_pipeline import ProductionFullPipeline
            pipeline = ProductionFullPipeline()
            
            # Run pipeline
            results = pipeline.run_full_pipeline()
            
            self.monitor.measure("ProductionFullPipeline completed")
            
            # Validate results
            if not isinstance(results, dict):
                self.errors.append("ProductionFullPipeline did not return dict")
                return {"status": "FAILED", "error": "Invalid return type"}
            
            if results.get("success", False):
                success("✅ ProductionFullPipeline test PASSED")
                return {"status": "SUCCESS", "results": results}
            else:
                warning("⚠️ ProductionFullPipeline completed with issues")
                return {"status": "PARTIAL", "results": results}
                
        except Exception as e:
            error(f"❌ ProductionFullPipeline test FAILED: {str(e)}")
            self.errors.append(f"ProductionFullPipeline error: {str(e)}")
            return {"status": "FAILED", "error": str(e)}
    
    def test_enhanced_pipeline(self) -> Dict[str, Any]:
        """ทดสอบ EnhancedFullPipeline"""
        try:
            info("🔍 Testing EnhancedFullPipeline...")
            self.monitor.measure("Starting EnhancedFullPipeline test")
            
            from enhanced_full_pipeline import EnhancedFullPipeline
            pipeline = EnhancedFullPipeline()
            
            # Run pipeline
            results = pipeline.run_full_pipeline()
            
            self.monitor.measure("EnhancedFullPipeline completed")
            
            # Validate results
            if not isinstance(results, dict):
                self.errors.append("EnhancedFullPipeline did not return dict")
                return {"status": "FAILED", "error": "Invalid return type"}
            
            if results.get("success", False):
                success("✅ EnhancedFullPipeline test PASSED")
                return {"status": "SUCCESS", "results": results}
            else:
                warning("⚠️ EnhancedFullPipeline completed with issues")
                return {"status": "PARTIAL", "results": results}
                
        except Exception as e:
            error(f"❌ EnhancedFullPipeline test FAILED: {str(e)}")
            self.errors.append(f"EnhancedFullPipeline error: {str(e)}")
            return {"status": "FAILED", "error": str(e)}
    
    def test_basic_pipeline(self) -> Dict[str, Any]:
        """ทดสอบ Basic Pipeline"""
        try:
            info("🔍 Testing Basic Pipeline...")
            self.monitor.measure("Starting Basic Pipeline test")
            
            from run_full_pipeline import run_full_pipeline

            # Run pipeline
            results = run_full_pipeline()
            
            self.monitor.measure("Basic Pipeline completed")
            
            # Validate results
            if results:
                success("✅ Basic Pipeline test PASSED")
                return {"status": "SUCCESS", "results": results}
            else:
                warning("⚠️ Basic Pipeline completed with issues")
                return {"status": "PARTIAL", "results": results}
                
        except Exception as e:
            error(f"❌ Basic Pipeline test FAILED: {str(e)}")
            self.errors.append(f"Basic Pipeline error: {str(e)}")
            return {"status": "FAILED", "error": str(e)}
    
    def test_all_pipelines(self) -> Dict[str, Any]:
        """ทดสอบ Pipeline ทั้งหมด"""
        info("🚀 Starting Comprehensive Full Pipeline Testing...")
        info("=" * 80)
        
        # Test order: most comprehensive first
        test_methods = [
            ("ComprehensiveProgressSystem", self.test_comprehensive_progress_system),
            ("ProductionFullPipeline", self.test_production_pipeline),
            ("EnhancedFullPipeline", self.test_enhanced_pipeline),
            ("BasicPipeline", self.test_basic_pipeline),
        ]
        
        for name, test_method in test_methods:
            info(f"\n📋 Testing {name}...")
            self.test_results[name] = test_method()
            
            # Force garbage collection
            gc.collect()
            
            # If successful, we can proceed to next or stop
            if self.test_results[name]["status"] == "SUCCESS":
                success(f"✅ {name} is working correctly!")
                break
            elif self.test_results[name]["status"] == "PARTIAL":
                warning(f"⚠️ {name} has issues but runs")
                # Continue to next test
            else:
                error(f"❌ {name} failed completely")
                # Continue to next test
        
        # Final summary
        self.monitor.measure("All tests completed")
        return self._generate_final_report()
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """สร้างรายงานสรุปผลการทดสอบ"""
        
        resource_summary = self.monitor.get_summary()
        
        # Count successful tests
        successful_tests = sum(1 for result in self.test_results.values() 
                             if result["status"] == "SUCCESS")
        partial_tests = sum(1 for result in self.test_results.values() 
                           if result["status"] == "PARTIAL")
        failed_tests = sum(1 for result in self.test_results.values() 
                          if result["status"] == "FAILED")
        
        # Determine overall status
        if successful_tests > 0:
            overall_status = "SUCCESS"
        elif partial_tests > 0:
            overall_status = "PARTIAL"
        else:
            overall_status = "FAILED"
        
        report = {
            "overall_status": overall_status,
            "test_summary": {
                "successful": successful_tests,
                "partial": partial_tests,
                "failed": failed_tests,
                "total": len(self.test_results)
            },
            "individual_results": self.test_results,
            "resource_usage": resource_summary,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": datetime.now().isoformat()
        }
        
        # Display summary
        info("\n" + "=" * 80)
        info("📊 FULL PIPELINE TEST SUMMARY")
        info("=" * 80)
        
        if overall_status == "SUCCESS":
            success(f"✅ Overall Status: {overall_status}")
        elif overall_status == "PARTIAL":
            warning(f"⚠️ Overall Status: {overall_status}")
        else:
            error(f"❌ Overall Status: {overall_status}")
        
        info(f"📈 Tests: {successful_tests} passed, {partial_tests} partial, {failed_tests} failed")
        info(f"⏱️ Total Time: {resource_summary['total_time']:.1f}s")
        info(f"🧠 Memory Usage: {resource_summary['memory_increase_mb']:.1f}MB increase")
        info(f"💻 Average CPU: {resource_summary['avg_cpu_percent']:.1f}%")
        
        if self.errors:
            error(f"❌ {len(self.errors)} errors found:")
            for err in self.errors:
                error(f"   • {err}")
        
        if self.warnings:
            warning(f"⚠️ {len(self.warnings)} warnings found:")
            for warn in self.warnings:
                warning(f"   • {warn}")
        
        return report


def main():
    """Main testing function"""
    print("\n🧪 COMPREHENSIVE FULL PIPELINE TESTING")
    print("=" * 80)
    
    validator = FullPipelineValidator()
    report = validator.test_all_pipelines()
    
    # Save report
    report_file = project_root / "FULL_PIPELINE_TEST_REPORT.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    success(f"📄 Report saved to: {report_file}")
    
    return report


if __name__ == "__main__":
    main()
