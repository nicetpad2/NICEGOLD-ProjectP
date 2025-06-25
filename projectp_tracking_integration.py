# Enterprise Tracking Integration for ProjectP
# projectp_tracking_integration.py
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from tracking import start_experiment, tracker
from tracking_integration import start_data_pipeline, start_production_monitoring
from typing import Dict, Any, Optional, List
import json
import logging
import os
import sys
"""
Integration script to connect the enterprise tracking system with existing ProjectP code
Handles automatic data issues logging and monitoring
"""


# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


console = Console()
logger = logging.getLogger(__name__)

class ProjectPTrackingIntegrator:
    """
    Integrates enterprise tracking with ProjectP ML pipeline
    """

    def __init__(self):
        self.current_experiment = None
        self.data_issues_log = []
        self.performance_metrics = {}

    def track_data_issues(self, issues: List[str], auto_fixes: Dict[str, Any] = None):
        """
        Track data issues and automatic fixes from ProjectP
        """
        with start_experiment("data_quality", f"data_issues_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as exp:

            # Log data issues
            exp.log_params({
                "total_issues": len(issues), 
                "issues_detected": ", ".join(issues[:5]),  # First 5 issues
                "auto_fix_enabled": auto_fixes is not None
            })

            # Log each issue type
            issue_counts = {}
            for issue in issues:
                issue_type = issue.split(':')[0] if ':' in issue else issue
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

            for issue_type, count in issue_counts.items():
                exp.log_metric(f"issue_count_{issue_type.replace(' ', '_')}", count)

            # Log auto - fixes if available
            if auto_fixes:
                exp.log_params(auto_fixes, prefix = "auto_fix_")
                exp.log_metric("auto_fixes_applied", len(auto_fixes))

            # Store for analysis
            self.data_issues_log.append({
                "timestamp": datetime.now().isoformat(), 
                "issues": issues, 
                "auto_fixes": auto_fixes or {}, 
                "run_id": exp.run_id
            })

            console.print(Panel(
                f"📊 Data Issues Tracked\n"
                f"Issues: {len(issues)}\n"
                f"Auto - fixes: {len(auto_fixes) if auto_fixes else 0}\n"
                f"Run ID: {exp.run_id}", 
                title = "Data Quality Tracking", 
                border_style = "yellow"
            ))

    def track_target_creation(self, target_info: Dict[str, Any]):
        """
        Track automatic target column creation
        """
        with start_experiment("feature_engineering", f"target_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as exp:

            exp.log_params({
                "target_column": target_info.get("column_name", "target"), 
                "creation_method": target_info.get("method", "median_split"), 
                "source_column": target_info.get("source", "Close"), 
                "threshold_value": target_info.get("threshold", "median")
            })

            # Calculate and log target statistics
            if "values" in target_info:
                values = target_info["values"]
                exp.log_metrics({
                    "target_mean": float(values.mean()) if hasattr(values, 'mean') else 0.0, 
                    "target_distribution_0": float((values == 0).sum()) if hasattr(values, 'sum') else 0.0, 
                    "target_distribution_1": float((values == 1).sum()) if hasattr(values, 'sum') else 0.0, 
                    "class_balance": float((values == 1).sum() / len(values)) if len(values) > 0 else 0.5
                })

            console.print(Panel(
                f"🎯 Target Creation Tracked\n"
                f"Method: {target_info.get('method', 'median_split')}\n"
                f"Source: {target_info.get('source', 'Close')}\n"
                f"Run ID: {exp.run_id}", 
                title = "Feature Engineering", 
                border_style = "green"
            ))

    def track_model_training(self, model_config: Dict[str, Any], results: Dict[str, Any]):
        """
        Track model training process and results
        """
        experiment_name = f"model_training_{model_config.get('model_type', 'unknown')}"
        run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with start_experiment(experiment_name, run_name) as exp:

            # Log model configuration
            exp.log_params(model_config, prefix = "model_")

            # Log training results
            if results:
                exp.log_metrics(results)

            # Log model if available
            if "model" in model_config:
                exp.log_model(model_config["model"], "trained_model")

            console.print(Panel(
                f"🤖 Model Training Tracked\n"
                f"Model: {model_config.get('model_type', 'Unknown')}\n"
                f"AUC: {results.get('auc', 'N/A')}\n"
                f"Run ID: {exp.run_id}", 
                title = "Model Training", 
                border_style = "blue"
            ))

    def track_pipeline_execution(self, pipeline_config: Dict[str, Any], performance: Dict[str, Any]):
        """
        Track full pipeline execution
        """
        with start_experiment("pipeline_execution", f"full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as exp:

            # Log pipeline configuration
            exp.log_params(pipeline_config, prefix = "pipeline_")

            # Log performance metrics
            exp.log_metrics(performance)

            # Log execution details
            exp.log_params({
                "execution_time": datetime.now().isoformat(), 
                "python_version": sys.version, 
                "working_directory": str(Path.cwd())
            })

            self.performance_metrics.update(performance)

            console.print(Panel(
                f"⚡ Pipeline Execution Tracked\n"
                f"Duration: {performance.get('duration_seconds', 'N/A')}s\n"
                f"Success: {performance.get('success', 'Unknown')}\n"
                f"Run ID: {exp.run_id}", 
                title = "Pipeline Tracking", 
                border_style = "purple"
            ))

    def start_production_monitoring_for_projectp(self, model_name: str = "projectp_model"):
        """
        Start production monitoring for ProjectP model
        """
        deployment_id = f"{model_name}_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_production_monitoring(model_name, deployment_id)

        console.print(Panel(
            f"🚀 Production Monitoring Started\n"
            f"Model: {model_name}\n"
            f"Deployment ID: {deployment_id}", 
            title = "Production Monitoring", 
            border_style = "red"
        ))

        return deployment_id

    def log_auc_improvement_attempt(self, attempt_info: Dict[str, Any]):
        """
        Track AUC improvement attempts
        """
        with start_experiment("auc_improvement", f"attempt_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as exp:

            exp.log_params(attempt_info, prefix = "attempt_")

            if "before_auc" in attempt_info and "after_auc" in attempt_info:
                improvement = attempt_info["after_auc"] - attempt_info["before_auc"]
                exp.log_metrics({
                    "auc_before": attempt_info["before_auc"], 
                    "auc_after": attempt_info["after_auc"], 
                    "auc_improvement": improvement, 
                    "improvement_success": 1 if improvement > 0 else 0
                })

            console.print(Panel(
                f"📈 AUC Improvement Tracked\n"
                f"Method: {attempt_info.get('method', 'Unknown')}\n"
                f"Before: {attempt_info.get('before_auc', 'N/A')}\n"
                f"After: {attempt_info.get('after_auc', 'N/A')}\n"
                f"Run ID: {exp.run_id}", 
                title = "AUC Improvement", 
                border_style = "cyan"
            ))

    def generate_project_report(self):
        """
        Generate comprehensive project report
        """
        report_data = {
            "generation_time": datetime.now().isoformat(), 
            "data_issues_log": self.data_issues_log, 
            "performance_metrics": self.performance_metrics, 
            "total_experiments": len(tracker.metadata) if hasattr(tracker, 'metadata') else 0
        }

        report_file = Path("reports") / f"project_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok = True)

        with open(report_file, 'w', encoding = 'utf - 8') as f:
            json.dump(report_data, f, indent = 2, default = str)

        console.print(Panel(
            f"📄 Project Report Generated\n"
            f"File: {report_file}\n"
            f"Data Issues: {len(self.data_issues_log)}\n"
            f"Experiments: {report_data['total_experiments']}", 
            title = "Report Generation", 
            border_style = "green"
        ))

        return str(report_file)

# Global integrator instance
projectp_integrator = ProjectPTrackingIntegrator()

# Convenience functions for ProjectP integration
def track_data_issues(issues: List[str], auto_fixes: Dict[str, Any] = None):
    """Track data issues from ProjectP warnings"""
    return projectp_integrator.track_data_issues(issues, auto_fixes)

def track_target_creation(target_info: Dict[str, Any]):
    """Track automatic target creation"""
    return projectp_integrator.track_target_creation(target_info)

def track_model_training(model_config: Dict[str, Any], results: Dict[str, Any]):
    """Track model training"""
    return projectp_integrator.track_model_training(model_config, results)

def track_pipeline_execution(pipeline_config: Dict[str, Any], performance: Dict[str, Any]):
    """Track pipeline execution"""
    return projectp_integrator.track_pipeline_execution(pipeline_config, performance)

def log_auc_improvement(attempt_info: Dict[str, Any]):
    """Log AUC improvement attempts"""
    return projectp_integrator.log_auc_improvement_attempt(attempt_info)

def start_projectp_monitoring(model_name: str = "projectp_model"):
    """Start production monitoring for ProjectP"""
    return projectp_integrator.start_production_monitoring_for_projectp(model_name)

def generate_project_report():
    """Generate project report"""
    return projectp_integrator.generate_project_report()

# Demo function to show integration
def demo_projectp_integration():
    """
    Demonstrate ProjectP integration with tracking system
    """
    console.print(Panel(
        "🧪 ProjectP Tracking Integration Demo", 
        title = "Demo", 
        border_style = "bold blue"
    ))

    # Simulate data issues (like the warnings you mentioned)
    data_issues = [
        "Missing target column: target", 
        "Data type inconsistency in Close column", 
        "Missing values in volume data"
    ]

    auto_fixes = {
        "target_creation": "median_split_from_Close", 
        "data_type_fix": "converted_to_float64", 
        "missing_value_handling": "forward_fill"
    }

    # Track data issues
    track_data_issues(data_issues, auto_fixes)

    # Track target creation
    target_info = {
        "column_name": "target", 
        "method": "median_split", 
        "source": "Close", 
        "threshold": "median"
    }

    track_target_creation(target_info)

    # Track model training
    model_config = {
        "model_type": "RandomForest", 
        "n_estimators": 100, 
        "max_depth": 10, 
        "random_state": 42
    }

    results = {
        "auc": 0.75, 
        "accuracy": 0.68, 
        "precision": 0.70, 
        "recall": 0.65
    }

    track_model_training(model_config, results)

    # Track AUC improvement attempt
    auc_attempt = {
        "method": "feature_engineering", 
        "before_auc": 0.65, 
        "after_auc": 0.75, 
        "technique": "technical_indicators"
    }

    log_auc_improvement(auc_attempt)

    # Track pipeline execution
    pipeline_config = {
        "data_source": "yahoo_finance", 
        "symbols": ["AAPL", "GOOGL", "MSFT"], 
        "timeframe": "1d", 
        "features": ["technical_indicators", "price_action"]
    }

    performance = {
        "duration_seconds": 145.2, 
        "records_processed": 10000, 
        "success": True, 
        "final_auc": 0.75
    }

    track_pipeline_execution(pipeline_config, performance)

    # Generate report
    report_file = generate_project_report()

    console.print(Panel(
        f"✅ ProjectP Integration Demo Complete!\n\n"
        f"All data issues and improvements are now tracked.\n"
        f"Report generated: {report_file}\n\n"
        f"To integrate with your existing code:\n"
        f"1. Import: from projectp_tracking_integration import track_data_issues\n"
        f"2. Use in your code when issues are detected\n"
        f"3. Monitor via: python tracking_cli.py list - experiments", 
        title = "Demo Complete", 
        border_style = "green"
    ))

if __name__ == "__main__":
    demo_projectp_integration()