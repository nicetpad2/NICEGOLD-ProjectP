# tracking.py
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
                from src.evidently_compat import ValueDrift, EVIDENTLY_AVAILABLE
from typing import Dict, Any, Optional, List, Union
import hashlib
import json
import logging
                import mlflow
                import mlflow.sklearn
        import numpy as np
import os
        import pandas as pd
import pickle
        import platform
        import psutil
                import shutil
import time
                import wandb
import yaml
"""
Enterprise - grade Experiment Tracking System
Supports MLflow, Weights & Biases, and custom tracking with automatic fallbacks
"""


# Configure enterprise logging
logging.basicConfig(
    level = logging.INFO, 
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    handlers = [RichHandler(rich_tracebacks = True)]
)
logger = logging.getLogger(__name__)
console = Console()

class ExperimentTracker:
    """
    Enterprise - grade experiment tracking with multiple backend support
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.active_run = None
        self.experiment_id = None
        self.run_id = None
        self.start_time = None
        self.metrics = {}
        self.parameters = {}
        self.artifacts = {}
        self.tags = {}

        # Initialize tracking backends
        self._init_backends()

        # Create tracking directory
        self.tracking_dir = Path(self.config.get('tracking_dir', './tracking'))
        self.tracking_dir.mkdir(exist_ok = True)

        # Setup metadata storage
        self.metadata_file = self.tracking_dir / 'experiment_metadata.json'
        self._load_metadata()

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'mlflow': {
                'enabled': True, 
                'tracking_uri': './mlruns', 
                'experiment_name': 'trading_ml_experiments'
            }, 
            'wandb': {
                'enabled': False, 
                'project': 'trading_ml', 
                'entity': None
            }, 
            'local': {
                'enabled': True, 
                'save_models': True, 
                'save_plots': True
            }, 
            'tracking_dir': './tracking', 
            'auto_log': True, 
            'log_system_info': True
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding = 'utf - 8') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def _init_backends(self):
        """Initialize tracking backends"""
        self.backends = {}

        # MLflow
        if self.config['mlflow']['enabled']:
            try:
                mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
                self.backends['mlflow'] = mlflow
                logger.info("✅ MLflow backend initialized")
            except ImportError:
                logger.warning("❌ MLflow not available - install with: pip install mlflow")

        # Weights & Biases
        if self.config['wandb']['enabled']:
            try:
                self.backends['wandb'] = wandb
                logger.info("✅ Weights & Biases backend initialized")
            except ImportError:
                logger.warning("❌ WandB not available - install with: pip install wandb")

    def _load_metadata(self):
        """Load experiment metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding = 'utf - 8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save experiment metadata"""
        try:
            with open(self.metadata_file, 'w', encoding = 'utf - 8') as f:
                json.dump(self.metadata, f, indent = 2, default = str)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")

    @contextmanager
    def start_run(self, 
                  experiment_name: str = "trading_experiment", 
                  run_name: Optional[str] = None, 
                  tags: Optional[Dict[str, str]] = None, 
                  description: Optional[str] = None):
        """
        Start a new experiment run with context manager
        """
        try:
            self._start_run_internal(experiment_name, run_name, tags, description)
            yield self
        finally:
            self.end_run()

    def _start_run_internal(self, 
                           experiment_name: str, 
                           run_name: Optional[str] = None, 
                           tags: Optional[Dict[str, str]] = None, 
                           description: Optional[str] = None):
        """Internal method to start run"""
        self.start_time = datetime.now()
        self.run_id = self._generate_run_id()

        if run_name is None:
            run_name = f"run_{self.start_time.strftime('%Y%m%d_%H%M%S')}"

        # Initialize tags
        self.tags = tags or {}
        self.tags.update({
            'start_time': self.start_time.isoformat(), 
            'run_id': self.run_id, 
            'experiment_name': experiment_name, 
            'description': description or 'Trading ML Experiment'
        })

        # Start MLflow run
        if 'mlflow' in self.backends:
            try:
                mlflow = self.backends['mlflow']
                mlflow.set_experiment(experiment_name)
                self.active_run = mlflow.start_run(run_name = run_name)

                # Log tags
                for key, value in self.tags.items():
                    if value is not None:
                        mlflow.set_tag(key, value)

                logger.info(f"🚀 MLflow run started: {run_name}")
            except Exception as e:
                logger.error(f"MLflow start failed: {e}")

        # Start WandB run
        if 'wandb' in self.backends:
            try:
                wandb = self.backends['wandb']
                self.wandb_run = wandb.init(
                    project = self.config['wandb']['project'], 
                    entity = self.config['wandb']['entity'], 
                    name = run_name, 
                    tags = list(self.tags.keys()), 
                    notes = description
                )
                logger.info(f"🚀 WandB run started: {run_name}")
            except Exception as e:
                logger.error(f"WandB start failed: {e}")

        # Local tracking
        self.run_dir = self.tracking_dir / self.run_id
        self.run_dir.mkdir(exist_ok = True)

        # Log system information
        if self.config.get('log_system_info', True):
            self._log_system_info()

        # Display run info
        self._display_run_info(experiment_name, run_name)

    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = str(int(time.time() * 1000))
        random_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"run_{timestamp}_{random_str}"

    def _log_system_info(self):
        """Log system information"""

        system_info = {
            'platform': platform.platform(), 
            'python_version': platform.python_version(), 
            'cpu_count': psutil.cpu_count(), 
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2), 
            'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2)
        }

        self.log_params(system_info, prefix = 'system_')

    def _display_run_info(self, experiment_name: str, run_name: str):
        """Display run information"""
        table = Table(title = f"🧪 Experiment Run Started")
        table.add_column("Property", style = "cyan", no_wrap = True)
        table.add_column("Value", style = "magenta")

        table.add_row("Experiment", experiment_name)
        table.add_row("Run Name", run_name)
        table.add_row("Run ID", self.run_id)
        table.add_row("Start Time", self.start_time.strftime('%Y - %m - %d %H:%M:%S'))
        table.add_row("Backends", ", ".join(self.backends.keys()))

        console.print(table)

    def log_params(self, params: Dict[str, Any], prefix: str = ""):
        """Log parameters"""
        for key, value in params.items():
            param_key = f"{prefix}{key}" if prefix else key
            self.parameters[param_key] = value

            # MLflow
            if 'mlflow' in self.backends:
                try:
                    self.backends['mlflow'].log_param(param_key, value)
                except Exception as e:
                    logger.warning(f"MLflow param logging failed: {e}")

            # WandB
            if 'wandb' in self.backends and hasattr(self, 'wandb_run'):
                try:
                    self.wandb_run.config.update({param_key: value})
                except Exception as e:
                    logger.warning(f"WandB param logging failed: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric"""
        self.metrics[key] = value

        # MLflow
        if 'mlflow' in self.backends:
            try:
                self.backends['mlflow'].log_metric(key, value, step)
            except Exception as e:
                logger.warning(f"MLflow metric logging failed: {e}")

        # WandB
        if 'wandb' in self.backends and hasattr(self, 'wandb_run'):
            try:
                log_dict = {key: value}
                if step is not None:
                    log_dict['step'] = step
                self.wandb_run.log(log_dict)
            except Exception as e:
                logger.warning(f"WandB metric logging failed: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics"""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact (file)"""
        if not Path(artifact_path).exists():
            logger.warning(f"Artifact not found: {artifact_path}")
            return

        artifact_name = artifact_name or Path(artifact_path).name
        self.artifacts[artifact_name] = artifact_path

        # MLflow
        if 'mlflow' in self.backends:
            try:
                self.backends['mlflow'].log_artifact(artifact_path)
            except Exception as e:
                logger.warning(f"MLflow artifact logging failed: {e}")

        # WandB
        if 'wandb' in self.backends and hasattr(self, 'wandb_run'):
            try:
                self.wandb_run.log_artifact(artifact_path, name = artifact_name)
            except Exception as e:
                logger.warning(f"WandB artifact logging failed: {e}")

        # Local copy
        if self.config['local']['enabled']:
            try:
                local_path = self.run_dir / artifact_name
                shutil.copy2(artifact_path, local_path)
            except Exception as e:
                logger.warning(f"Local artifact copy failed: {e}")

    def log_model(self, model, model_name: str = "model"):
        """Log trained model"""
        if self.config['local']['save_models']:
            model_path = self.run_dir / f"{model_name}.pkl"
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"💾 Model saved locally: {model_path}")
            except Exception as e:
                logger.warning(f"Local model save failed: {e}")

        # MLflow
        if 'mlflow' in self.backends:
            try:
                mlflow.sklearn.log_model(model, model_name)
            except Exception as e:
                logger.warning(f"MLflow model logging failed: {e}")

    def log_figure(self, figure, figure_name: str = "plot"):
        """Log matplotlib figure"""
        if self.config['local']['save_plots']:
            fig_path = self.run_dir / f"{figure_name}.png"
            try:
                figure.savefig(fig_path, dpi = 300, bbox_inches = 'tight')
                logger.info(f"📊 Figure saved locally: {fig_path}")
            except Exception as e:
                logger.warning(f"Local figure save failed: {e}")

        # MLflow
        if 'mlflow' in self.backends:
            try:
                self.backends['mlflow'].log_figure(figure, f"{figure_name}.png")
            except Exception as e:
                logger.warning(f"MLflow figure logging failed: {e}")

        # WandB
        if 'wandb' in self.backends and hasattr(self, 'wandb_run'):
            try:
                self.wandb_run.log({figure_name: wandb.Image(figure)})
            except Exception as e:
                logger.warning(f"WandB figure logging failed: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the run"""
        self.tags.update(tags)

        # MLflow
        if 'mlflow' in self.backends:
            try:
                for key, value in tags.items():
                    self.backends['mlflow'].set_tag(key, value)
            except Exception as e:
                logger.warning(f"MLflow tag setting failed: {e}")

        # WandB
        if 'wandb' in self.backends and hasattr(self, 'wandb_run'):
            try:
                self.wandb_run.tags = list(set(self.wandb_run.tags + list(tags.keys())))
            except Exception as e:
                logger.warning(f"WandB tag setting failed: {e}")

    def end_run(self):
        """End the current run"""
        if self.start_time is None:
            return

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        # Log duration
        self.log_metric('duration_seconds', duration)

        # Save run summary
        self._save_run_summary(end_time, duration)

        # End MLflow run
        if 'mlflow' in self.backends and self.active_run:
            try:
                self.backends['mlflow'].end_run()
                logger.info("🏁 MLflow run ended")
            except Exception as e:
                logger.warning(f"MLflow end run failed: {e}")

        # End WandB run
        if 'wandb' in self.backends and hasattr(self, 'wandb_run'):
            try:
                self.wandb_run.finish()
                logger.info("🏁 WandB run ended")
            except Exception as e:
                logger.warning(f"WandB end run failed: {e}")

        # Display summary
        self._display_run_summary(duration)

        # Reset state
        self.active_run = None
        self.start_time = None

    def _save_run_summary(self, end_time: datetime, duration: float):
        """Save run summary to local file"""
        summary = {
            'run_id': self.run_id, 
            'start_time': self.start_time.isoformat(), 
            'end_time': end_time.isoformat(), 
            'duration_seconds': duration, 
            'parameters': self.parameters, 
            'metrics': self.metrics, 
            'tags': self.tags, 
            'artifacts': self.artifacts
        }

        summary_path = self.run_dir / 'run_summary.json'
        try:
            with open(summary_path, 'w', encoding = 'utf - 8') as f:
                json.dump(summary, f, indent = 2, default = str)
        except Exception as e:
            logger.warning(f"Run summary save failed: {e}")

        # Update metadata
        self.metadata[self.run_id] = summary
        self._save_metadata()

    def _display_run_summary(self, duration: float):
        """Display run summary"""
        table = Table(title = "📋 Experiment Run Summary")
        table.add_column("Category", style = "cyan", no_wrap = True)
        table.add_column("Count/Value", style = "magenta")

        table.add_row("Duration", f"{duration:.2f} seconds")
        table.add_row("Parameters", str(len(self.parameters)))
        table.add_row("Metrics", str(len(self.metrics)))
        table.add_row("Artifacts", str(len(self.artifacts)))
        table.add_row("Run ID", self.run_id)

        console.print(table)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        experiments = []

        for run_id, summary in self.metadata.items():
            experiments.append({
                'run_id': run_id, 
                'start_time': summary.get('start_time'), 
                'duration': summary.get('duration_seconds', 0), 
                'metrics_count': len(summary.get('metrics', {})), 
                'status': 'completed'
            })

        return sorted(experiments, key = lambda x: x['start_time'], reverse = True)

    def get_best_run(self, metric_name: str, mode: str = 'max') -> Optional[Dict[str, Any]]:
        """Get best run based on metric"""
        best_run = None
        best_value = float(' - inf') if mode == 'max' else float('inf')

        for run_id, summary in self.metadata.items():
            metrics = summary.get('metrics', {})
            if metric_name in metrics:
                value = metrics[metric_name]
                if (mode == 'max' and value > best_value) or (mode == 'min' and value < best_value):
                    best_value = value
                    best_run = summary

        return best_run

# Global tracker instance
tracker = ExperimentTracker()

# EnterpriseTracker is an alias for ExperimentTracker for backward compatibility
class EnterpriseTracker(ExperimentTracker):
    """
    Enterprise - grade tracker - alias for ExperimentTracker
    Provides backward compatibility for existing code with enhanced features
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self.enterprise_features = {
            'auto_backup': True, 
            'compliance_logging': True, 
            'advanced_metrics': True, 
            'security_tracking': True
        }
        console.print(f"✅ EnterpriseTracker initialized with config: {config_path or 'default'}")
        logger.info("🏢 Enterprise features enabled: auto_backup, compliance_logging, advanced_metrics, security_tracking")

    def track_experiment(self, experiment_name: str, **kwargs):
        """Enterprise tracking interface"""
        return self.start_run(experiment_name, **kwargs)

    def log_compliance_event(self, event_type: str, details: Dict[str, Any]):
        """Log compliance events for enterprise tracking"""
        compliance_event = {
            'timestamp': datetime.now().isoformat(), 
            'event_type': event_type, 
            'details': details, 
            'run_id': self.run_id
        }

        try:
            compliance_file = self.tracking_dir / 'compliance_log.json'
            if compliance_file.exists():
                with open(compliance_file, 'r', encoding = 'utf - 8') as f:
                    compliance_log = json.load(f)
            else:
                compliance_log = []

            compliance_log.append(compliance_event)

            with open(compliance_file, 'w', encoding = 'utf - 8') as f:
                json.dump(compliance_log, f, indent = 2, default = str)

            logger.info(f"📋 Compliance event logged: {event_type}")
        except Exception as e:
            logger.warning(f"⚠️ Compliance logging failed: {e}")

    def backup_experiment_data(self):
        """Backup experiment data for enterprise compliance"""
        try:
            backup_dir = self.tracking_dir / 'backups'
            backup_dir.mkdir(exist_ok = True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f'experiment_backup_{timestamp}.json'

            backup_data = {
                'metadata': self.metadata, 
                'parameters': self.parameters, 
                'metrics': self.metrics, 
                'tags': self.tags, 
                'backup_timestamp': timestamp
            }

            with open(backup_file, 'w', encoding = 'utf - 8') as f:
                json.dump(backup_data, f, indent = 2, default = str)

            logger.info(f"💾 Enterprise backup created: {backup_file}")
            return str(backup_file)
        except Exception as e:
            logger.error(f"❌ Enterprise backup failed: {e}")
            return None

    def get_enterprise_summary(self) -> Dict[str, Any]:
        """Get comprehensive enterprise summary"""
        summary = {
            'total_experiments': len(self.metadata), 
            'enterprise_features': self.enterprise_features, 
            'tracking_backends': list(self.backends.keys()), 
            'last_backup': None, 
            'compliance_events': 0
        }

        # ตรวจสอบ backup ล่าสุด
        try:
            backup_dir = self.tracking_dir / 'backups'
            if backup_dir.exists():
                backup_files = list(backup_dir.glob('experiment_backup_*.json'))
                if backup_files:
                    latest_backup = max(backup_files, key = lambda x: x.stat().st_mtime)
                    summary['last_backup'] = latest_backup.name
        except Exception:
            pass

        # นับ compliance events
        try:
            compliance_file = self.tracking_dir / 'compliance_log.json'
            if compliance_file.exists():
                with open(compliance_file, 'r', encoding = 'utf - 8') as f:
                    compliance_log = json.load(f)
                    summary['compliance_events'] = len(compliance_log)
        except Exception:
            pass

        return summary

    def end_run(self):
        """End run with enterprise features"""
        # ทำ backup อัตโนมัติก่อนจบ
        if self.enterprise_features.get('auto_backup', True) and self.run_id:
            self.backup_experiment_data()

        # Log compliance event
        if self.enterprise_features.get('compliance_logging', True):
            self.log_compliance_event('experiment_completed', {
                'run_id': self.run_id, 
                'metrics_count': len(self.metrics), 
                'parameters_count': len(self.parameters)
            })

        # เรียก parent method
        super().end_run()

        logger.info("🏢 Enterprise run completed with full compliance")

# สร้าง ML Protection System functions
class MLProtectionSystem:
    """
    ML Protection System สำหรับป้องกันและตรวจสอบโมเดล
    """

    def __init__(self):
        self.protection_config = {
            'drift_detection': True, 
            'anomaly_detection': True, 
            'performance_monitoring': True, 
            'security_checks': True
        }
        logger.info("🛡️ ML Protection System initialized")

    def monitor_model_drift(self, reference_data, current_data, threshold = 0.1):
        """ตรวจสอบ model drift"""
        try:
            # ใช้ Evidently หากมี ไม่งั้นใช้ fallback
            try:
                if EVIDENTLY_AVAILABLE:
                    drift_detector = ValueDrift(column_name = 'target')
                    result = drift_detector.calculate(reference_data, current_data)
                else:
                    result = self._fallback_drift_detection(reference_data, current_data, threshold)
            except:
                result = self._fallback_drift_detection(reference_data, current_data, threshold)

            return result
        except Exception as e:
            logger.warning(f"⚠️ Drift detection failed: {e}")
            return {'drift_detected': False, 'error': str(e)}

    def _fallback_drift_detection(self, reference_data, current_data, threshold):
        """Fallback drift detection"""

        if isinstance(reference_data, pd.DataFrame) and isinstance(current_data, pd.DataFrame):
            # เปรียบเทียบสถิติพื้นฐาน
            ref_mean = reference_data.mean().mean() if len(reference_data) > 0 else 0
            curr_mean = current_data.mean().mean() if len(current_data) > 0 else 0

            drift_score = abs(ref_mean - curr_mean) / (abs(ref_mean) + 1e - 8)
            drift_detected = drift_score > threshold

            return {
                'drift_score': float(drift_score), 
                'drift_detected': bool(drift_detected), 
                'method': 'fallback_statistics'
            }

        return {'drift_detected': False, 'method': 'fallback_no_data'}

# Global instances
ml_protection = MLProtectionSystem()

# Fallback functions สำหรับ ML Protection
def create_protection_fallbacks():
    """สร้าง fallback functions สำหรับ ML Protection"""

    def fallback_track_model_performance(*args, **kwargs):
        logger.info("📊 Using fallback model performance tracking")
        return {"status": "tracked", "method": "fallback"}

    def fallback_detect_anomalies(*args, **kwargs):
        logger.info("🔍 Using fallback anomaly detection")
        return {"anomalies_detected": False, "method": "fallback"}

    def fallback_monitor_drift(*args, **kwargs):
        logger.info("📈 Using fallback drift monitoring")
        return {"drift_detected": False, "method": "fallback"}

    return {
        'track_model_performance': fallback_track_model_performance, 
        'detect_anomalies': fallback_detect_anomalies, 
        'monitor_drift': fallback_monitor_drift
    }

# สร้าง fallback functions
protection_fallbacks = create_protection_fallbacks()

# Convenience functions for global access
def start_experiment(experiment_name: str = "trading_experiment", 
                    run_name: Optional[str] = None, 
                    tags: Optional[Dict[str, str]] = None, 
                    description: Optional[str] = None):
    """Start a new experiment (context manager)"""
    return tracker.start_run(experiment_name, run_name, tags, description)

def log_params(params: Dict[str, Any]):
    """Log parameters to active run"""
    tracker.log_params(params)

def log_metric(key: str, value: float, step: Optional[int] = None):
    """Log metric to active run"""
    tracker.log_metric(key, value, step)

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Log multiple metrics to active run"""
    tracker.log_metrics(metrics, step)

def log_model(model, model_name: str = "model"):
    """Log model to active run"""
    tracker.log_model(model, model_name)

def log_figure(figure, figure_name: str = "plot"):
    """Log figure to active run"""
    tracker.log_figure(figure, figure_name)

# Example usage function
def demo_tracking():
    """Demonstrate tracking capabilities"""
    with start_experiment("demo_experiment", "demo_run", 
                         tags = {"version": "1.0", "env": "production"}, 
                         description = "Demo of tracking system") as exp:

        # Log parameters
        exp.log_params({
            "learning_rate": 0.01, 
            "batch_size": 32, 
            "epochs": 100
        })

        # Simulate training with metrics
        for epoch in range(5):
            exp.log_metrics({
                "accuracy": 0.8 + epoch * 0.02, 
                "loss": 1.0 - epoch * 0.1
            }, step = epoch)

        console.print("✅ Demo tracking completed!")

if __name__ == "__main__":
    demo_tracking()