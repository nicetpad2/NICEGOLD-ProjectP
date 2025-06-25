#!/usr/bin/env python3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import json
import os
import psutil
import threading
import time
"""
🔧 RESOURCE MANAGEMENT UTILITIES
Advanced resource monitoring and control for NICEGOLD Pipeline
"""

@dataclass
class ResourceThresholds:
    """Resource usage thresholds"""

    max_cpu_percent: float = 80.0
    max_ram_percent: float = 80.0
    max_disk_usage_percent: float = 90.0
    warning_cpu_percent: float = 70.0
    warning_ram_percent: float = 70.0
    critical_cpu_percent: float = 90.0
    critical_ram_percent: float = 90.0

@dataclass
class ResourceSnapshot:
    """Resource usage snapshot"""

    timestamp: datetime
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    process_count: int
    cpu_temp: Optional[float] = None

class ResourceMonitorAdvanced:
    """Advanced resource monitor with predictive capabilities"""

    def __init__(self, thresholds: ResourceThresholds = None):
        self.thresholds = thresholds or ResourceThresholds()
        self.snapshots: List[ResourceSnapshot] = []
        self.alerts: List[Dict] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current resource usage snapshot"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval = 1)

        # Memory usage
        memory = psutil.virtual_memory()
        ram_percent = memory.percent
        ram_used_gb = memory.used / (1024**3)
        ram_total_gb = memory.total / (1024**3)

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)

        # Process count
        process_count = len(psutil.pids())

        # CPU temperature (if available)
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps:
                cpu_temp = temps["coretemp"][0].current
        except:
            pass

        return ResourceSnapshot(
            timestamp = datetime.now(), 
            cpu_percent = cpu_percent, 
            ram_percent = ram_percent, 
            ram_used_gb = ram_used_gb, 
            ram_total_gb = ram_total_gb, 
            disk_usage_percent = disk_usage_percent, 
            disk_free_gb = disk_free_gb, 
            process_count = process_count, 
            cpu_temp = cpu_temp, 
        )

    def check_resource_status(self, snapshot: ResourceSnapshot) -> Dict[str, str]:
        """Check resource status against thresholds"""
        status = {"overall": "OK", "cpu": "OK", "ram": "OK", "disk": "OK"}

        # CPU status
        if snapshot.cpu_percent >= self.thresholds.critical_cpu_percent:
            status["cpu"] = "CRITICAL"
            status["overall"] = "CRITICAL"
        elif snapshot.cpu_percent >= self.thresholds.max_cpu_percent:
            status["cpu"] = "HIGH"
            if status["overall"] == "OK":
                status["overall"] = "HIGH"
        elif snapshot.cpu_percent >= self.thresholds.warning_cpu_percent:
            status["cpu"] = "WARNING"
            if status["overall"] == "OK":
                status["overall"] = "WARNING"

        # RAM status
        if snapshot.ram_percent >= self.thresholds.critical_ram_percent:
            status["ram"] = "CRITICAL"
            status["overall"] = "CRITICAL"
        elif snapshot.ram_percent >= self.thresholds.max_ram_percent:
            status["ram"] = "HIGH"
            if status["overall"] in ["OK", "WARNING"]:
                status["overall"] = "HIGH"
        elif snapshot.ram_percent >= self.thresholds.warning_ram_percent:
            status["ram"] = "WARNING"
            if status["overall"] == "OK":
                status["overall"] = "WARNING"

        # Disk status
        if snapshot.disk_usage_percent >= self.thresholds.max_disk_usage_percent:
            status["disk"] = "HIGH"
            if status["overall"] in ["OK", "WARNING"]:
                status["overall"] = "HIGH"

        return status

    def predict_resource_trend(self, minutes_ahead: int = 5) -> Dict[str, float]:
        """Predict resource usage trend"""
        if len(self.snapshots) < 3:
            return {"cpu_trend": 0, "ram_trend": 0}

        # Simple linear trend calculation
        recent_snapshots = self.snapshots[ - 5:]  # Last 5 snapshots

        # CPU trend
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        cpu_trend = (cpu_values[ - 1] - cpu_values[0]) / len(cpu_values)

        # RAM trend
        ram_values = [s.ram_percent for s in recent_snapshots]
        ram_trend = (ram_values[ - 1] - ram_values[0]) / len(ram_values)

        # Predict future values
        predicted_cpu = recent_snapshots[ - 1].cpu_percent + (cpu_trend * minutes_ahead)
        predicted_ram = recent_snapshots[ - 1].ram_percent + (ram_trend * minutes_ahead)

        return {
            "cpu_trend": cpu_trend, 
            "ram_trend": ram_trend, 
            "predicted_cpu": max(0, min(100, predicted_cpu)), 
            "predicted_ram": max(0, min(100, predicted_ram)), 
        }

    def get_resource_recommendations(self) -> List[str]:
        """Get resource optimization recommendations"""
        if not self.snapshots:
            return ["No resource data available for recommendations"]

        latest = self.snapshots[ - 1]
        status = self.check_resource_status(latest)
        recommendations = []

        if status["cpu"] in ["HIGH", "CRITICAL"]:
            recommendations.extend(
                [
                    "🔴 High CPU usage detected:", 
                    "  • Consider reducing parallel processing", 
                    "  • Check for CPU - intensive operations", 
                    "  • Monitor background processes", 
                ]
            )

        if status["ram"] in ["HIGH", "CRITICAL"]:
            recommendations.extend(
                [
                    "🔴 High RAM usage detected:", 
                    "  • Clear unused data from memory", 
                    "  • Consider processing data in smaller batches", 
                    "  • Check for memory leaks", 
                ]
            )

        if status["disk"] == "HIGH":
            recommendations.extend(
                [
                    "🔴 Low disk space:", 
                    "  • Clean temporary files", 
                    "  • Archive old output files", 
                    "  • Check output directory size", 
                ]
            )

        # Performance recommendations
        if latest.process_count > 200:
            recommendations.append("⚠️ High process count - consider system cleanup")

        if len(recommendations) == 0:
            recommendations.append("✅ System resources are healthy")

        return recommendations

    def start_monitoring(self, interval_seconds: int = 5):
        """Start continuous resource monitoring"""
        if self.monitoring:
            return

        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                try:
                    snapshot = self.get_current_snapshot()
                    self.snapshots.append(snapshot)

                    # Keep only last 100 snapshots
                    if len(self.snapshots) > 100:
                        self.snapshots = self.snapshots[ - 100:]

                    # Check for alerts
                    status = self.check_resource_status(snapshot)
                    if status["overall"] in ["HIGH", "CRITICAL"]:
                        alert = {
                            "timestamp": snapshot.timestamp.isoformat(), 
                            "level": status["overall"], 
                            "cpu_percent": snapshot.cpu_percent, 
                            "ram_percent": snapshot.ram_percent, 
                            "message": f"Resource usage {status['overall']}: CPU {snapshot.cpu_percent:.1f}%, RAM {snapshot.ram_percent:.1f}%", 
                        }
                        self.alerts.append(alert)

                    time.sleep(interval_seconds)

                except Exception as e:
                    print(f"Resource monitoring error: {e}")
                    time.sleep(interval_seconds)

        self.monitor_thread = threading.Thread(target = monitor_loop, daemon = True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout = 5)

    def export_resource_report(self, output_file: str):
        """Export resource monitoring report"""
        if not self.snapshots:
            return

        report = {
            "monitoring_period": {
                "start": self.snapshots[0].timestamp.isoformat(), 
                "end": self.snapshots[ - 1].timestamp.isoformat(), 
                "duration_minutes": (
                    self.snapshots[ - 1].timestamp - self.snapshots[0].timestamp
                ).total_seconds()
                / 60, 
            }, 
            "resource_summary": {
                "avg_cpu_percent": sum(s.cpu_percent for s in self.snapshots)
                / len(self.snapshots), 
                "max_cpu_percent": max(s.cpu_percent for s in self.snapshots), 
                "avg_ram_percent": sum(s.ram_percent for s in self.snapshots)
                / len(self.snapshots), 
                "max_ram_percent": max(s.ram_percent for s in self.snapshots), 
                "min_disk_free_gb": min(s.disk_free_gb for s in self.snapshots), 
            }, 
            "alerts": self.alerts, 
            "recommendations": self.get_resource_recommendations(), 
            "snapshots": [
                {
                    "timestamp": s.timestamp.isoformat(), 
                    "cpu_percent": s.cpu_percent, 
                    "ram_percent": s.ram_percent, 
                    "ram_used_gb": s.ram_used_gb, 
                    "disk_free_gb": s.disk_free_gb, 
                }
                for s in self.snapshots[ - 20:]  # Last 20 snapshots
            ], 
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent = 2)

class ResourceController:
    """Control pipeline execution based on resource availability"""

    def __init__(self, monitor: ResourceMonitorAdvanced):
        self.monitor = monitor
        self.pause_callbacks: List[Callable] = []
        self.resume_callbacks: List[Callable] = []

    def add_pause_callback(self, callback: Callable):
        """Add callback to execute when pipeline should pause"""
        self.pause_callbacks.append(callback)

    def add_resume_callback(self, callback: Callable):
        """Add callback to execute when pipeline can resume"""
        self.resume_callbacks.append(callback)

    def should_pause_pipeline(self) -> bool:
        """Check if pipeline should be paused due to resource constraints"""
        if not self.monitor.snapshots:
            return False

        latest = self.monitor.snapshots[ - 1]
        status = self.monitor.check_resource_status(latest)

        # Pause if resources are critical
        if status["overall"] == "CRITICAL":
            return True

        # Pause if predicted usage will be critical
        trend = self.monitor.predict_resource_trend(2)  # 2 minutes ahead
        if (
            trend["predicted_cpu"] > self.monitor.thresholds.critical_cpu_percent
            or trend["predicted_ram"] > self.monitor.thresholds.critical_ram_percent
        ):
            return True

        return False

    def wait_for_resources(self, max_wait_minutes: int = 10) -> bool:
        """Wait for resources to become available"""
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60

        while time.time() - start_time < max_wait_seconds:
            if not self.should_pause_pipeline():
                return True

            # Execute pause callbacks
            for callback in self.pause_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Pause callback error: {e}")

            time.sleep(10)  # Check every 10 seconds

        return False

    def optimize_for_stage(self, stage_name: str) -> Dict[str, Any]:
        """Get optimization settings for specific pipeline stage"""
        latest = self.monitor.snapshots[ - 1] if self.monitor.snapshots else None

        if not latest:
            return {"batch_size": 1000, "n_jobs": 1, "memory_limit": "4GB"}

        # Adjust settings based on current resource usage
        cpu_available = 100 - latest.cpu_percent
        ram_available_gb = (100 - latest.ram_percent) / 100 * latest.ram_total_gb

        # Calculate optimal settings
        if cpu_available > 50 and ram_available_gb > 8:
            # High resources available
            settings = {
                "batch_size": 5000, 
                "n_jobs": -1,  # Use all CPUs
                "memory_limit": f"{int(ram_available_gb * 0.7)}GB", 
            }
        elif cpu_available > 30 and ram_available_gb > 4:
            # Medium resources available
            settings = {
                "batch_size": 2000, 
                "n_jobs": max(1, int(psutil.cpu_count() * 0.7)), 
                "memory_limit": f"{int(ram_available_gb * 0.8)}GB", 
            }
        else:
            # Low resources available
            settings = {
                "batch_size": 500, 
                "n_jobs": 1, 
                "memory_limit": f"{int(ram_available_gb * 0.9)}GB", 
            }

        # Stage - specific adjustments
        if stage_name in ["Train", "WalkForward"]:
            # ML stages need more memory
            settings["memory_limit"] = (
                f'{int(float(settings["memory_limit"].replace("GB", "")) * 1.2)}GB'
            )
        elif stage_name in ["Preprocess", "Features"]:
            # Data processing stages can use more CPU
            settings["n_jobs"] = min(settings["n_jobs"] * 2, psutil.cpu_count())

        return settings

def create_resource_manager() -> tuple[ResourceMonitorAdvanced, ResourceController]:
    """Create configured resource manager components"""
    # Set conservative thresholds for production
    thresholds = ResourceThresholds(
        max_cpu_percent = 75.0,  # Conservative 75% instead of 80%
        max_ram_percent = 75.0, 
        warning_cpu_percent = 60.0, 
        warning_ram_percent = 60.0, 
        critical_cpu_percent = 85.0, 
        critical_ram_percent = 85.0, 
    )

    monitor = ResourceMonitorAdvanced(thresholds)
    controller = ResourceController(monitor)

    return monitor, controller

if __name__ == "__main__":
    # Test resource monitoring
    monitor, controller = create_resource_manager()

    print("🔍 Testing resource monitoring...")

    # Start monitoring
    monitor.start_monitoring(interval_seconds = 2)

    # Monitor for 30 seconds
    time.sleep(30)

    # Stop monitoring
    monitor.stop_monitoring()

    # Export report
    monitor.export_resource_report("test_resource_report.json")

    print("✅ Resource monitoring test completed")
    print(f"📊 Collected {len(monitor.snapshots)} snapshots")
    print(f"⚠️ Generated {len(monitor.alerts)} alerts")

    # Show recommendations
    recommendations = monitor.get_resource_recommendations()
    print("\n💡 Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")