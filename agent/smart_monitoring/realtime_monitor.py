from collections import deque, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Dict, List, Any, Optional, Callable
import json
import os
import psutil
import threading
import time
                import yaml
"""
Real - time Monitor
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Real - time monitoring of ProjectP pipeline performance and health metrics.
"""


class RealtimeMonitor:
    """
    Real - time monitoring system for continuous project health assessment.
    """

    def __init__(self, project_root: str, monitoring_interval: float = 5.0):
        self.project_root = Path(project_root)
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None

        # Monitoring data storage
        self.metrics_history = deque(maxlen = 1000)
        self.alerts_queue = Queue()
        self.performance_baseline = {}
        self.thresholds = self._load_monitoring_thresholds()

        # Event callbacks
        self.event_callbacks = defaultdict(list)

        # Metrics tracking
        self.current_metrics = {}
        self.trend_analyzer = TrendAnalyzer()

    def start_monitoring(self) -> None:
        """Start real - time monitoring."""
        if self.is_monitoring:
            print("⚠️ Monitoring already running")
            return

        print("🔍 Starting real - time monitoring...")
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target = self._monitoring_loop, daemon = True)
        self.monitor_thread.start()

        # Establish baseline if not exists
        if not self.performance_baseline:
            self._establish_baseline()

    def stop_monitoring(self) -> None:
        """Stop real - time monitoring."""
        print("⏹️ Stopping real - time monitoring...")
        self.is_monitoring = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout = 10)

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect current metrics
                current_metrics = self._collect_metrics()

                # Store metrics with timestamp
                timestamped_metrics = {
                    'timestamp': datetime.now().isoformat(), 
                    'metrics': current_metrics
                }
                self.metrics_history.append(timestamped_metrics)
                self.current_metrics = current_metrics

                # Analyze trends and detect anomalies
                self._analyze_trends()
                self._detect_anomalies(current_metrics)

                # Check thresholds and generate alerts
                self._check_thresholds(current_metrics)

                # Trigger callbacks
                self._trigger_callbacks('metrics_updated', current_metrics)

                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)

            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system and project metrics."""
        metrics = {}

        try:
            # System metrics
            process = psutil.Process()
            system_mem = psutil.virtual_memory()

            metrics['system'] = {
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024, 
                'memory_percent': process.memory_percent(), 
                'cpu_percent': process.cpu_percent(), 
                'system_memory_percent': system_mem.percent, 
                'system_memory_available_gb': system_mem.available / 1024 / 1024 / 1024, 
                'thread_count': process.num_threads()
            }

            # Project - specific metrics
            metrics['project'] = {
                'file_count': self._count_project_files(), 
                'code_quality_score': self._calculate_code_quality_score(), 
                'last_modification_time': self._get_last_modification_time(), 
                'error_rate': self._calculate_error_rate(), 
                'performance_score': self._calculate_performance_score()
            }

            # Pipeline metrics (if available)
            metrics['pipeline'] = {
                'last_run_time': self._get_last_pipeline_run_time(), 
                'success_rate': self._calculate_pipeline_success_rate(), 
                'average_execution_time': self._get_average_execution_time(), 
                'auc_trend': self._get_auc_trend()
            }

            # Health metrics
            metrics['health'] = {
                'overall_health_score': self._calculate_overall_health_score(metrics), 
                'critical_issues_count': self._count_critical_issues(), 
                'dependency_health': self._check_dependency_health(), 
                'data_quality_score': self._assess_data_quality()
            }

        except Exception as e:
            print(f"⚠️ Error collecting metrics: {e}")

        return metrics

    def _analyze_trends(self) -> None:
        """Analyze trends in collected metrics."""
        if len(self.metrics_history) < 10:
            return  # Need at least 10 data points

        try:
            # Get recent metrics for trend analysis
            recent_metrics = list(self.metrics_history)[ - 10:]

            # Analyze memory trend
            memory_values = [m['metrics']['system']['memory_usage_mb'] for m in recent_metrics]
            memory_trend = self.trend_analyzer.analyze_trend(memory_values)

            # Analyze performance trend
            if 'performance_score' in recent_metrics[ - 1]['metrics']['project']:
                perf_values = [m['metrics']['project']['performance_score'] for m in recent_metrics]
                perf_trend = self.trend_analyzer.analyze_trend(perf_values)

                # Store trend information
                self.current_metrics['trends'] = {
                    'memory_trend': memory_trend, 
                    'performance_trend': perf_trend
                }

        except Exception as e:
            print(f"⚠️ Trend analysis error: {e}")

    def _detect_anomalies(self, current_metrics: Dict[str, Any]) -> None:
        """Detect anomalies in current metrics."""
        try:
            anomalies = []

            # Memory anomaly detection
            memory_usage = current_metrics['system']['memory_usage_mb']
            if self.performance_baseline.get('memory_baseline'):
                baseline_memory = self.performance_baseline['memory_baseline']
                if memory_usage > baseline_memory * 1.5:  # 50% above baseline
                    anomalies.append({
                        'type': 'memory_spike', 
                        'severity': 'high', 
                        'current': memory_usage, 
                        'baseline': baseline_memory, 
                        'message': f"Memory usage {memory_usage:.1f}MB is 50% above baseline {baseline_memory:.1f}MB"
                    })

            # Performance anomaly detection
            if 'performance_score' in current_metrics['project']:
                perf_score = current_metrics['project']['performance_score']
                if self.performance_baseline.get('performance_baseline'):
                    baseline_perf = self.performance_baseline['performance_baseline']
                    if perf_score < baseline_perf * 0.7:  # 30% below baseline
                        anomalies.append({
                            'type': 'performance_degradation', 
                            'severity': 'medium', 
                            'current': perf_score, 
                            'baseline': baseline_perf, 
                            'message': f"Performance score {perf_score:.3f} is 30% below baseline {baseline_perf:.3f}"
                        })

            # Health score anomaly
            health_score = current_metrics['health']['overall_health_score']
            if health_score < 0.5:
                anomalies.append({
                    'type': 'health_degradation', 
                    'severity': 'high', 
                    'current': health_score, 
                    'message': f"Overall health score {health_score:.3f} is critically low"
                })

            # Store anomalies
            if anomalies:
                self.current_metrics['anomalies'] = anomalies
                self._trigger_callbacks('anomaly_detected', anomalies)

        except Exception as e:
            print(f"⚠️ Anomaly detection error: {e}")

    def _check_thresholds(self, current_metrics: Dict[str, Any]) -> None:
        """Check metrics against configured thresholds."""
        try:
            alerts = []

            for category, thresholds in self.thresholds.items():
                if category in current_metrics:
                    for metric_name, threshold_config in thresholds.items():
                        if metric_name in current_metrics[category]:
                            current_value = current_metrics[category][metric_name]

                            # Check threshold breach
                            if self._is_threshold_breached(current_value, threshold_config):
                                alert = {
                                    'type': 'threshold_breach', 
                                    'category': category, 
                                    'metric': metric_name, 
                                    'current_value': current_value, 
                                    'threshold': threshold_config, 
                                    'timestamp': datetime.now().isoformat(), 
                                    'severity': threshold_config.get('severity', 'medium')
                                }
                                alerts.append(alert)
                                self.alerts_queue.put(alert)

            if alerts:
                self._trigger_callbacks('threshold_breach', alerts)

        except Exception as e:
            print(f"⚠️ Threshold checking error: {e}")

    def _establish_baseline(self) -> None:
        """Establish performance baseline."""
        print("📊 Establishing performance baseline...")

        try:
            # Collect baseline metrics over several cycles
            baseline_metrics = []
            for _ in range(5):
                metrics = self._collect_metrics()
                baseline_metrics.append(metrics)
                time.sleep(2)

            # Calculate baseline values
            memory_values = [m['system']['memory_usage_mb'] for m in baseline_metrics]

            self.performance_baseline = {
                'memory_baseline': sum(memory_values) / len(memory_values), 
                'established_at': datetime.now().isoformat()
            }

            # Add performance baseline if available
            perf_values = [m['project'].get('performance_score', 0.5) for m in baseline_metrics]
            if any(v > 0 for v in perf_values):
                self.performance_baseline['performance_baseline'] = sum(perf_values) / len(perf_values)

            print(f"✅ Baseline established: Memory = {self.performance_baseline['memory_baseline']:.1f}MB")

        except Exception as e:
            print(f"⚠️ Failed to establish baseline: {e}")

    def _load_monitoring_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Load monitoring thresholds from configuration."""
        default_thresholds = {
            'system': {
                'memory_usage_mb': {'max': 2048, 'severity': 'high'}, 
                'memory_percent': {'max': 80, 'severity': 'medium'}, 
                'cpu_percent': {'max': 90, 'severity': 'high'}, 
                'system_memory_percent': {'max': 85, 'severity': 'critical'}
            }, 
            'project': {
                'code_quality_score': {'min': 0.6, 'severity': 'medium'}, 
                'error_rate': {'max': 0.1, 'severity': 'high'}, 
                'performance_score': {'min': 0.5, 'severity': 'medium'}
            }, 
            'health': {
                'overall_health_score': {'min': 0.6, 'severity': 'high'}, 
                'critical_issues_count': {'max': 5, 'severity': 'critical'}
            }
        }

        # Try to load from config file
        config_path = self.project_root / 'agent_config.yaml'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                if 'monitoring' in config and 'thresholds' in config['monitoring']:
                    # Merge with defaults
                    thresholds = default_thresholds.copy()
                    thresholds.update(config['monitoring']['thresholds'])
                    return thresholds

            except Exception as e:
                print(f"⚠️ Failed to load monitoring config: {e}")

        return default_thresholds

    def _is_threshold_breached(self, value: float, threshold_config: Dict[str, Any]) -> bool:
        """Check if a value breaches the threshold."""
        if 'max' in threshold_config and value > threshold_config['max']:
            return True
        if 'min' in threshold_config and value < threshold_config['min']:
            return True
        return False

    def _count_project_files(self) -> int:
        """Count total project files."""
        try:
            return len(list(self.project_root.rglob("*.py")))
        except:
            return 0

    def _calculate_code_quality_score(self) -> float:
        """Calculate basic code quality score."""
        try:
            # Simple heuristic based on file structure and patterns
            python_files = list(self.project_root.rglob("*.py"))
            if not python_files:
                return 0.0

            score = 0.7  # Base score

            # Check for good practices
            has_tests = any('test' in str(f).lower() for f in python_files)
            has_docs = (self.project_root / 'README.md').exists()
            has_requirements = (self.project_root / 'requirements.txt').exists()

            if has_tests:
                score += 0.1
            if has_docs:
                score += 0.1
            if has_requirements:
                score += 0.1

            return min(1.0, score)

        except:
            return 0.5

    def _get_last_modification_time(self) -> Optional[str]:
        """Get last modification time of project files."""
        try:
            python_files = list(self.project_root.rglob("*.py"))
            if not python_files:
                return None

            last_modified = max(f.stat().st_mtime for f in python_files)
            return datetime.fromtimestamp(last_modified).isoformat()

        except:
            return None

    def _calculate_error_rate(self) -> float:
        """Calculate error rate based on logs or file patterns."""
        try:
            # Look for error patterns in recent logs
            log_files = list(self.project_root.rglob("*.log"))
            error_count = 0
            total_lines = 0

            for log_file in log_files[:5]:  # Check last 5 log files
                try:
                    with open(log_file, 'r', encoding = 'utf - 8', errors = 'ignore') as f:
                        lines = f.readlines()[ - 100:]  # Last 100 lines
                        total_lines += len(lines)
                        error_count += sum(1 for line in lines if any(
                            keyword in line.lower() for keyword in ['error', 'exception', 'failed']
                        ))
                except:
                    continue

            return error_count / max(1, total_lines)

        except:
            return 0.0

    def _calculate_performance_score(self) -> float:
        """Calculate performance score."""
        try:
            # Simple heuristic based on system metrics
            process = psutil.Process()
            memory_percent = process.memory_percent()
            cpu_percent = process.cpu_percent()

            # Higher resource usage = lower performance score
            performance_score = 1.0 - (memory_percent / 100 * 0.5 + cpu_percent / 100 * 0.5)
            return max(0.0, min(1.0, performance_score))

        except:
            return 0.5

    def _get_last_pipeline_run_time(self) -> Optional[str]:
        """Get last pipeline run time."""
        try:
            # Check for pipeline output files or logs
            output_files = list(self.project_root.rglob("output*"))
            if output_files:
                latest_file = max(output_files, key = lambda f: f.stat().st_mtime)
                return datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
        except:
            pass
        return None

    def _calculate_pipeline_success_rate(self) -> float:
        """Calculate pipeline success rate."""
        # Placeholder implementation
        return 0.85  # Default success rate

    def _get_average_execution_time(self) -> float:
        """Get average pipeline execution time."""
        # Placeholder implementation
        return 120.0  # Default 2 minutes

    def _get_auc_trend(self) -> Dict[str, Any]:
        """Get AUC trend information."""
        return {
            'current_auc': 0.65,  # Placeholder
            'trend': 'improving', 
            'change_rate': 0.02
        }

    def _calculate_overall_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall health score."""
        try:
            scores = []

            # System health component
            memory_score = 1.0 - min(1.0, metrics['system']['memory_percent'] / 100)
            cpu_score = 1.0 - min(1.0, metrics['system']['cpu_percent'] / 100)
            system_score = (memory_score + cpu_score) / 2
            scores.append(system_score * 0.3)

            # Project health component
            project_score = metrics['project']['code_quality_score']
            scores.append(project_score * 0.4)

            # Performance component
            perf_score = metrics['project']['performance_score']
            scores.append(perf_score * 0.3)

            return sum(scores)

        except:
            return 0.5

    def _count_critical_issues(self) -> int:
        """Count critical issues in the project."""
        # Placeholder implementation
        return 0

    def _check_dependency_health(self) -> float:
        """Check dependency health."""
        try:
            # Check if requirements.txt exists and is readable
            req_file = self.project_root / 'requirements.txt'
            if req_file.exists():
                return 0.8
            else:
                return 0.5
        except:
            return 0.3

    def _assess_data_quality(self) -> float:
        """Assess data quality score."""
        try:
            # Look for data files and assess basic quality
            data_files = list(self.project_root.rglob("*.csv")) + list(self.project_root.rglob("*.parquet"))
            if data_files:
                return 0.75  # Assume reasonable quality if data files exist
            else:
                return 0.5   # Neutral score if no data files
        except:
            return 0.5

    def add_callback(self, event_type: str, callback: Callable) -> None:
        """Add callback for monitoring events."""
        self.event_callbacks[event_type].append(callback)

    def remove_callback(self, event_type: str, callback: Callable) -> None:
        """Remove callback for monitoring events."""
        if callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)

    def _trigger_callbacks(self, event_type: str, data: Any) -> None:
        """Trigger callbacks for a specific event type."""
        for callback in self.event_callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                print(f"⚠️ Callback error for {event_type}: {e}")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status and metrics."""
        return {
            'is_monitoring': self.is_monitoring, 
            'current_metrics': self.current_metrics, 
            'alerts_pending': self.alerts_queue.qsize(), 
            'metrics_history_size': len(self.metrics_history), 
            'baseline_established': bool(self.performance_baseline)
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {'message': 'No metrics collected yet'}

        recent_metrics = list(self.metrics_history)[ - 10:]  # Last 10 entries

        # Calculate averages
        avg_memory = sum(m['metrics']['system']['memory_usage_mb'] for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m['metrics']['system']['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_health = sum(m['metrics']['health']['overall_health_score'] for m in recent_metrics) / len(recent_metrics)

        return {
            'average_memory_usage_mb': avg_memory, 
            'average_cpu_percent': avg_cpu, 
            'average_health_score': avg_health, 
            'metrics_collected': len(self.metrics_history), 
            'monitoring_duration_minutes': len(self.metrics_history) * self.monitoring_interval / 60
        }


class TrendAnalyzer:
    """Helper class for trend analysis."""

    def analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in a series of values."""
        if len(values) < 3:
            return {'trend': 'insufficient_data'}

        try:
            # Simple linear trend analysis
            n = len(values)
            x = list(range(n))

            # Calculate slope
            x_mean = sum(x) / n
            y_mean = sum(values) / n

            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator

            # Classify trend
            if abs(slope) < 0.01:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'

            return {
                'trend': trend, 
                'slope': slope, 
                'strength': abs(slope), 
                'last_value': values[ - 1], 
                'change_from_start': values[ - 1] - values[0]
            }

        except Exception as e:
            return {'trend': 'error', 'error': str(e)}