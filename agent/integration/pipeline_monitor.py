"""
Pipeline Monitor System
ระบบ monitoring pipeline แบบ real-time
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
import logging
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class PipelineMonitor:
    """ระบบ monitoring pipeline แบบ real-time"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics = {}
        self.alerts = []
        self.thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 85.0,
            'execution_time': 3600,  # 1 hour
            'error_rate': 0.1
        }
        
    def start_monitoring(self, interval: float = 5.0):
        """เริ่ม monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Pipeline monitoring started")
    
    def stop_monitoring(self):
        """หยุด monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
        logger.info("Pipeline monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Loop สำหรับ monitoring"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Check thresholds
                self._check_thresholds(metrics)
                
                # Store metrics
                timestamp = datetime.now().isoformat()
                self.metrics[timestamp] = metrics
                
                # Cleanup old metrics (keep last 1000 entries)
                if len(self.metrics) > 1000:
                    oldest_keys = sorted(self.metrics.keys())[:100]
                    for key in oldest_keys:
                        del self.metrics[key]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """เก็บ system metrics"""
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
            'process_count': len(psutil.pids()),
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """ตรวจสอบ thresholds และสร้าง alerts"""
        for metric, threshold in self.thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                alert = {
                    'type': 'threshold_exceeded',
                    'metric': metric,
                    'value': metrics[metric],
                    'threshold': threshold,
                    'timestamp': datetime.now().isoformat(),
                    'severity': self._get_severity(metric, metrics[metric], threshold)
                }
                self.alerts.append(alert)
                logger.warning(f"Threshold exceeded: {metric} = {metrics[metric]} > {threshold}")
    
    def _get_severity(self, metric: str, value: float, threshold: float) -> str:
        """กำหนด severity ของ alert"""
        ratio = value / threshold
        if ratio > 1.5:
            return 'critical'
        elif ratio > 1.2:
            return 'high'
        else:
            return 'medium'
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """ดึง metrics ปัจจุบัน"""
        if not self.metrics:
            return {}
        
        latest_timestamp = max(self.metrics.keys())
        return self.metrics[latest_timestamp]
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """สรุป metrics ย้อนหลัง N ชั่วโมง"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()
        
        recent_metrics = {
            k: v for k, v in self.metrics.items() 
            if k >= cutoff_str
        }
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        cpu_values = [m['cpu_usage'] for m in recent_metrics.values()]
        memory_values = [m['memory_usage'] for m in recent_metrics.values()]
        
        return {
            'period_hours': hours,
            'data_points': len(recent_metrics),
            'cpu_avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            'cpu_max': max(cpu_values) if cpu_values else 0,
            'memory_avg': sum(memory_values) / len(memory_values) if memory_values else 0,
            'memory_max': max(memory_values) if memory_values else 0,
            'alerts_count': len([a for a in self.alerts if a['timestamp'] >= cutoff_str])
        }
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """ดึง alerts ที่ยังไม่ได้แก้ไข"""
        alerts = self.alerts.copy()
        
        if severity:
            alerts = [a for a in alerts if a.get('severity') == severity]
        
        # Return last 50 alerts
        return alerts[-50:]
    
    def clear_alerts(self):
        """เคลียร์ alerts"""
        self.alerts.clear()
        logger.info("Alerts cleared")
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """อัพเดท thresholds"""
        self.thresholds.update(new_thresholds)
        logger.info(f"Thresholds updated: {new_thresholds}")
    
    def monitor_pipeline_execution(self, func: Callable, *args, **kwargs):
        """Monitor การ execute function"""
        start_time = time.time()
        start_metrics = self._collect_system_metrics()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            end_metrics = self._collect_system_metrics()
            execution_time = end_time - start_time
            
            # Store execution metrics
            execution_data = {
                'function': func.__name__ if hasattr(func, '__name__') else str(func),
                'execution_time': execution_time,
                'success': success,
                'error': error,
                'start_metrics': start_metrics,
                'end_metrics': end_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in metrics with special key
            exec_key = f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.metrics[exec_key] = execution_data
            
            # Check execution time threshold
            if execution_time > self.thresholds.get('execution_time', 3600):
                alert = {
                    'type': 'long_execution',
                    'function': execution_data['function'],
                    'execution_time': execution_time,
                    'threshold': self.thresholds['execution_time'],
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'medium'
                }
                self.alerts.append(alert)
        
        return result
    
    def save_monitoring_report(self, filepath: Optional[str] = None) -> str:
        """บันทึกรายงาน monitoring"""
        if not filepath:
            filepath = os.path.join(self.project_root, "monitoring_report.json")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'total_metrics': len(self.metrics),
            'total_alerts': len(self.alerts),
            'current_metrics': self.get_current_metrics(),
            'metrics_summary_1h': self.get_metrics_summary(1),
            'active_alerts': self.get_active_alerts(),
            'thresholds': self.thresholds
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Monitoring report saved to {filepath}")
        return filepath
