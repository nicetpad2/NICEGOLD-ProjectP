# Enterprise Tracking System Integration
# tracking_integration.py
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TaskID
from tracking import ExperimentTracker, tracker
from typing import Dict, Any, Optional, List, Callable
import hashlib
import json
import logging
import os
        import psutil
import queue
import threading
import time
"""
Integration utilities for enterprise tracking system
Handles data pipeline integration, model deployment tracking, and production monitoring
"""


console = Console()
logger = logging.getLogger(__name__)

class ProductionTracker:
    """
    Production - grade tracking for live trading systems
    """

    def __init__(self, config_path: Optional[str] = None):
        self.tracker = ExperimentTracker(config_path)
        self.production_runs = {}
        self.monitoring_active = False
        self.alert_queue = queue.Queue()
        self.metrics_buffer = {}
        self.performance_monitor = None

    def start_production_monitoring(self, model_name: str, deployment_id: str):
        """Start production monitoring for deployed model"""
        with self.tracker.start_run(
            experiment_name = "production_monitoring", 
            run_name = f"{model_name}_deployment_{deployment_id}", 
            tags = {
                "environment": "production", 
                "model_name": model_name, 
                "deployment_id": deployment_id, 
                "monitoring": "active"
            }, 
            description = "Production model monitoring and performance tracking"
        ) as run:

            self.production_runs[deployment_id] = {
                "model_name": model_name, 
                "start_time": datetime.now(), 
                "run": run, 
                "metrics": {}, 
                "alerts": []
            }

            # Start performance monitoring thread
            self.monitoring_active = True
            self.performance_monitor = threading.Thread(
                target = self._monitor_performance, 
                args = (deployment_id, ), 
                daemon = True
            )
            self.performance_monitor.start()

            console.print(Panel(
                f"🚀 Production monitoring started for {model_name}", 
                title = "Production Tracker", 
                border_style = "green"
            ))

    def _monitor_performance(self, deployment_id: str):
        """Monitor system performance in background"""

        while self.monitoring_active:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval = 1)
                memory_percent = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent

                # Log to production run
                if deployment_id in self.production_runs:
                    run = self.production_runs[deployment_id]["run"]
                    timestamp = int(time.time())

                    run.log_metrics({
                        "system_cpu_percent": cpu_percent, 
                        "system_memory_percent": memory_percent, 
                        "system_disk_percent": disk_usage, 
                    }, step = timestamp)

                    # Alert on high resource usage
                    if cpu_percent > 90 or memory_percent > 90:
                        self._add_alert(deployment_id, "high_resource_usage", {
                            "cpu": cpu_percent, 
                            "memory": memory_percent
                        })

                time.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(10)

    def log_prediction(self, deployment_id: str, 
                      input_data: Dict[str, Any], 
                      prediction: Any, 
                      confidence: Optional[float] = None, 
                      latency_ms: Optional[float] = None):
        """Log individual prediction for monitoring"""
        if deployment_id not in self.production_runs:
            logger.warning(f"No active monitoring for deployment {deployment_id}")
            return

        run = self.production_runs[deployment_id]["run"]
        timestamp = int(time.time())

        # Log prediction metrics
        metrics = {
            "prediction_count": 1, 
        }

        if confidence is not None:
            metrics["prediction_confidence"] = confidence

        if latency_ms is not None:
            metrics["prediction_latency_ms"] = latency_ms

            # Alert on high latency
            if latency_ms > 1000:  # 1 second threshold
                self._add_alert(deployment_id, "high_latency", {
                    "latency_ms": latency_ms
                })

        run.log_metrics(metrics, step = timestamp)

        # Store prediction data (optional - for debugging)
        prediction_data = {
            "timestamp": timestamp, 
            "input_hash": hashlib.md5(str(input_data).encode()).hexdigest()[:8], 
            "prediction": str(prediction), 
            "confidence": confidence, 
            "latency_ms": latency_ms
        }

        # Save to local file for detailed analysis
        pred_file = Path(run.run_dir) / "predictions.jsonl"
        with open(pred_file, 'a', encoding = 'utf - 8') as f:
            f.write(json.dumps(prediction_data) + '\n')

    def log_trade_result(self, deployment_id: str, 
                        trade_data: Dict[str, Any]):
        """Log trading results for performance evaluation"""
        if deployment_id not in self.production_runs:
            return

        run = self.production_runs[deployment_id]["run"]
        timestamp = int(time.time())

        # Extract trading metrics
        metrics = {}
        for key in ['pnl', 'return_pct', 'win_rate', 'sharpe_ratio']:
            if key in trade_data:
                metrics[f"trade_{key}"] = trade_data[key]

        run.log_metrics(metrics, step = timestamp)

        # Alert on significant losses
        if 'pnl' in trade_data and trade_data['pnl'] < -10000:  # Threshold
            self._add_alert(deployment_id, "significant_loss", {
                "pnl": trade_data['pnl']
            })

    def _add_alert(self, deployment_id: str, alert_type: str, data: Dict[str, Any]):
        """Add alert to queue"""
        alert = {
            "timestamp": datetime.now().isoformat(), 
            "deployment_id": deployment_id, 
            "type": alert_type, 
            "data": data
        }

        self.alert_queue.put(alert)
        self.production_runs[deployment_id]["alerts"].append(alert)

        logger.warning(f"🚨 Alert: {alert_type} for {deployment_id}")

    def get_production_summary(self, deployment_id: str) -> Dict[str, Any]:
        """Get production monitoring summary"""
        if deployment_id not in self.production_runs:
            return {}

        run_data = self.production_runs[deployment_id]
        duration = datetime.now() - run_data["start_time"]

        return {
            "model_name": run_data["model_name"], 
            "deployment_id": deployment_id, 
            "uptime_hours": duration.total_seconds() / 3600, 
            "alert_count": len(run_data["alerts"]), 
            "last_alert": run_data["alerts"][ - 1] if run_data["alerts"] else None, 
            "status": "active" if self.monitoring_active else "stopped"
        }

    def stop_monitoring(self, deployment_id: str):
        """Stop production monitoring"""
        if deployment_id in self.production_runs:
            self.monitoring_active = False
            if self.performance_monitor:
                self.performance_monitor.join(timeout = 5)

            # End the tracking run
            run_data = self.production_runs[deployment_id]
            duration = datetime.now() - run_data["start_time"]

            run_data["run"].log_metric("total_uptime_hours", duration.total_seconds() / 3600)
            run_data["run"].log_metric("total_alerts", len(run_data["alerts"]))

            console.print(Panel(
                f"🛑 Production monitoring stopped for {run_data['model_name']}", 
                title = "Production Tracker", 
                border_style = "red"
            ))

class DataPipelineTracker:
    """
    Track data pipeline operations and data quality
    """

    def __init__(self):
        self.pipeline_runs = {}

    def start_pipeline_run(self, pipeline_name: str, 
                          data_source: str, 
                          expected_records: Optional[int] = None):
        """Start tracking data pipeline run"""
        with tracker.start_run(
            experiment_name = "data_pipeline", 
            run_name = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
            tags = {
                "pipeline_name": pipeline_name, 
                "data_source": data_source, 
                "type": "data_pipeline"
            }
        ) as run:

            run.log_params({
                "pipeline_name": pipeline_name, 
                "data_source": data_source, 
                "expected_records": expected_records
            })

            return DataPipelineRun(run, pipeline_name)

class DataPipelineRun:
    """
    Individual data pipeline run context
    """

    def __init__(self, tracker_run, pipeline_name: str):
        self.run = tracker_run
        self.pipeline_name = pipeline_name
        self.start_time = datetime.now()
        self.stages = []

    def log_stage(self, stage_name: str, 
                  records_processed: int, 
                  errors: int = 0, 
                  duration_seconds: Optional[float] = None):
        """Log pipeline stage completion"""
        stage_data = {
            "stage_name": stage_name, 
            "records_processed": records_processed, 
            "errors": errors, 
            "duration_seconds": duration_seconds, 
            "timestamp": datetime.now().isoformat()
        }

        self.stages.append(stage_data)

        # Log metrics
        self.run.log_metrics({
            f"stage_{stage_name}_records": records_processed, 
            f"stage_{stage_name}_errors": errors, 
            f"stage_{stage_name}_duration": duration_seconds or 0
        })

        logger.info(f"📊 Pipeline stage completed: {stage_name} ({records_processed} records)")

    def log_data_quality(self, quality_metrics: Dict[str, float]):
        """Log data quality metrics"""
        for metric, value in quality_metrics.items():
            self.run.log_metric(f"data_quality_{metric}", value)

    def complete_pipeline(self, total_records: int, success: bool = True):
        """Complete pipeline tracking"""
        duration = (datetime.now() - self.start_time).total_seconds()

        self.run.log_metrics({
            "total_records_processed": total_records, 
            "total_duration_seconds": duration, 
            "pipeline_success": 1 if success else 0, 
            "stages_completed": len(self.stages)
        })

        # Save detailed stage information
        stages_file = Path(self.run.run_dir) / "pipeline_stages.json"
        with open(stages_file, 'w', encoding = 'utf - 8') as f:
            json.dump(self.stages, f, indent = 2)

        status = "✅ SUCCESS" if success else "❌ FAILED"
        console.print(Panel(
            f"{status} Pipeline: {self.pipeline_name}\n"
            f"Duration: {duration:.2f}s\n"
            f"Records: {total_records:, }\n"
            f"Stages: {len(self.stages)}", 
            title = "Data Pipeline", 
            border_style = "green" if success else "red"
        ))

class ModelDeploymentTracker:
    """
    Track model deployment and A/B testing
    """

    def __init__(self):
        self.deployments = {}

    def deploy_model(self, model_name: str, 
                    model_version: str, 
                    deployment_config: Dict[str, Any]):
        """Track model deployment"""
        deployment_id = f"{model_name}_v{model_version}_{int(time.time())}"

        with tracker.start_run(
            experiment_name = "model_deployment", 
            run_name = f"deploy_{deployment_id}", 
            tags = {
                "model_name": model_name, 
                "model_version": model_version, 
                "deployment_type": deployment_config.get("type", "production"), 
                "environment": deployment_config.get("environment", "production")
            }
        ) as run:

            run.log_params(deployment_config)

            # Log deployment success
            run.log_metric("deployment_success", 1)
            run.log_metric("deployment_timestamp", time.time())

            self.deployments[deployment_id] = {
                "model_name": model_name, 
                "model_version": model_version, 
                "deployment_time": datetime.now(), 
                "config": deployment_config, 
                "run": run
            }

            console.print(Panel(
                f"🚀 Model deployed: {model_name} v{model_version}\n"
                f"Deployment ID: {deployment_id}", 
                title = "Model Deployment", 
                border_style = "green"
            ))

            return deployment_id

# Global instances
production_tracker = ProductionTracker()
pipeline_tracker = DataPipelineTracker()
deployment_tracker = ModelDeploymentTracker()

# Convenience functions
def start_production_monitoring(model_name: str, deployment_id: str):
    """Start production monitoring"""
    return production_tracker.start_production_monitoring(model_name, deployment_id)

def log_prediction(deployment_id: str, input_data: Dict, prediction: Any, 
                  confidence: float = None, latency_ms: float = None):
    """Log prediction"""
    return production_tracker.log_prediction(
        deployment_id, input_data, prediction, confidence, latency_ms
    )

def start_data_pipeline(pipeline_name: str, data_source: str, expected_records: int = None):
    """Start data pipeline tracking"""
    return pipeline_tracker.start_pipeline_run(pipeline_name, data_source, expected_records)

def deploy_model(model_name: str, model_version: str, config: Dict[str, Any]):
    """Deploy and track model"""
    return deployment_tracker.deploy_model(model_name, model_version, config)

if __name__ == "__main__":
    # Demo usage
    console.print("🧪 Testing Enterprise Tracking Integration...")

    # Test data pipeline tracking
    with start_data_pipeline("market_data_ingestion", "yahoo_finance", 1000) as pipeline:
        pipeline.log_stage("data_fetch", 950, 0, 5.2)
        pipeline.log_stage("data_clean", 940, 10, 2.1)
        pipeline.log_stage("feature_engineering", 940, 0, 8.5)
        pipeline.log_data_quality({"completeness": 0.95, "accuracy": 0.98})
        pipeline.complete_pipeline(940, success = True)

    console.print("✅ Integration test completed!")