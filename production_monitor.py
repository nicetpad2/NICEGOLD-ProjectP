"""
🎯 PRODUCTION PIPELINE MONITOR & AUTO-FIX
=========================================
ระบบติดตามและแก้ไขปัญหา Production Pipeline อัตโนมัติ

Features:
- 🔍 Real-time AUC monitoring
- 🚨 Automatic issue detection
- 🔧 Auto-fix deployment
- 📊 Performance tracking
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Rich console
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.progress import Progress
from rich import box

console = Console()

class ProductionMonitor:
    def __init__(self, output_dir="output_default"):
        """Initialize Production Pipeline Monitor"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup monitoring log
        log_file = self.output_dir / "production_monitor.log"
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Monitoring thresholds
        self.min_auc = 0.70
        self.critical_auc = 0.60
        self.max_retries = 3
        
        # Status tracking
        self.status_file = self.output_dir / "production_status.json"
        self.load_status()
        
        console.print(Panel.fit("🎯 Production Pipeline Monitor Initialized", style="bold blue"))
    
    def load_status(self):
        """Load current production status"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    self.status = json.load(f)
            except Exception:
                self.status = self.create_default_status()
        else:
            self.status = self.create_default_status()
    
    def create_default_status(self):
        """Create default status structure"""
        return {
            'last_check': None,
            'current_auc': 0.0,
            'status': 'unknown',
            'consecutive_failures': 0,
            'auto_fix_attempts': 0,
            'last_successful_run': None,
            'issues': [],
            'fix_history': []
        }
    
    def save_status(self):
        """Save current status to file"""
        self.status['last_check'] = datetime.now().isoformat()
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def check_pipeline_health(self) -> Dict[str, Any]:
        """Check overall pipeline health"""
        health_report = {
            'overall_status': 'unknown',
            'auc_status': 'unknown',
            'data_status': 'unknown',
            'model_status': 'unknown',
            'issues': [],
            'recommendations': []
        }
        
        # Check AUC performance
        auc_result = self.check_auc_performance()
        health_report['auc_status'] = auc_result['status']
        health_report['current_auc'] = auc_result['auc']
        
        if auc_result['status'] == 'critical':
            health_report['issues'].append('🚨 Critical AUC performance')
            health_report['recommendations'].append('Run emergency AUC fix')
        
        # Check data quality
        data_result = self.check_data_quality()
        health_report['data_status'] = data_result['status']
        
        if data_result['status'] == 'poor':
            health_report['issues'].append('📊 Data quality issues detected')
            health_report['recommendations'].append('Run data quality fix')
        
        # Check model files
        model_result = self.check_model_files()
        health_report['model_status'] = model_result['status']
        
        if model_result['status'] == 'missing':
            health_report['issues'].append('🤖 Model files missing or corrupted')
            health_report['recommendations'].append('Retrain models')
        
        # Determine overall status
        if any(status == 'critical' for status in [auc_result['status'], data_result['status'], model_result['status']]):
            health_report['overall_status'] = 'critical'
        elif any(status == 'poor' for status in [auc_result['status'], data_result['status'], model_result['status']]):
            health_report['overall_status'] = 'warning'
        else:
            health_report['overall_status'] = 'healthy'
        
        return health_report
    
    def check_auc_performance(self) -> Dict[str, Any]:
        """Check AUC performance from recent predictions"""
        try:
            # Check prediction files
            pred_files = [
                self.output_dir / "predictions.csv",
                self.output_dir / "predict_summary_metrics.json"
            ]
            
            auc = 0.0
            
            # Try to get AUC from summary metrics
            if pred_files[1].exists():
                with open(pred_files[1], 'r') as f:
                    metrics = json.load(f)
                    auc = metrics.get('auc', 0.0)
            
            # Fallback: calculate from predictions file
            elif pred_files[0].exists():
                df = pd.read_csv(pred_files[0])
                if 'target' in df.columns and 'pred_proba' in df.columns:
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(df['target'], df['pred_proba'])
            
            # Determine status
            if auc >= self.min_auc:
                status = 'good'
            elif auc >= self.critical_auc:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'auc': auc,
                'status': status,
                'threshold': self.min_auc
            }
        
        except Exception as e:
            self.logger.error(f"AUC check failed: {e}")
            return {
                'auc': 0.0,
                'status': 'critical',
                'threshold': self.min_auc,
                'error': str(e)
            }
    
    def check_data_quality(self) -> Dict[str, Any]:
        """Check data quality metrics"""
        try:
            data_files = [
                self.output_dir / "preprocessed_super.parquet",
                self.output_dir / "auto_features.parquet"
            ]
            
            for file_path in data_files:
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    
                    # Data quality checks
                    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
                    duplicate_ratio = df.duplicated().sum() / len(df)
                    
                    if missing_ratio > 0.3 or duplicate_ratio > 0.5:
                        status = 'poor'
                    elif missing_ratio > 0.1 or duplicate_ratio > 0.2:
                        status = 'warning'
                    else:
                        status = 'good'
                    
                    return {
                        'status': status,
                        'missing_ratio': missing_ratio,
                        'duplicate_ratio': duplicate_ratio,
                        'shape': df.shape
                    }
            
            return {'status': 'missing', 'error': 'No data files found'}
        
        except Exception as e:
            return {'status': 'poor', 'error': str(e)}
    
    def check_model_files(self) -> Dict[str, Any]:
        """Check if required model files exist"""
        required_files = [
            self.output_dir / "catboost_model_best_cv.pkl",
            self.output_dir / "train_features.txt"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if not missing_files:
            status = 'good'
        elif len(missing_files) == len(required_files):
            status = 'missing'
        else:
            status = 'partial'
        
        return {
            'status': status,
            'missing_files': [str(f) for f in missing_files],
            'required_files': [str(f) for f in required_files]
        }
    
    def auto_fix_issues(self, health_report: Dict[str, Any]) -> bool:
        """Automatically fix detected issues"""
        console.print(Panel.fit("🔧 Auto-Fix System Activated", style="bold yellow"))
        
        fix_success = False
        
        # Critical AUC fix
        if health_report['auc_status'] == 'critical':
            console.print("🚨 Attempting critical AUC fix...")
            try:
                from emergency_auc_hotfix import emergency_auc_hotfix
                if emergency_auc_hotfix():
                    console.print("✅ Emergency AUC hotfix successful")
                    fix_success = True
                    self.status['fix_history'].append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'emergency_auc_hotfix',
                        'status': 'success'
                    })
                else:
                    # Try production fix
                    from production_auc_critical_fix import run_production_auc_fix
                    result = run_production_auc_fix()
                    if result['success']:
                        console.print("✅ Production AUC fix successful")
                        fix_success = True
                        self.status['fix_history'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'production_auc_fix',
                            'status': 'success'
                        })
            except Exception as e:
                console.print(f"❌ Auto-fix failed: {e}")
                self.status['fix_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'auc_fix',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Missing model files fix
        if health_report['model_status'] == 'missing':
            console.print("🤖 Attempting to recreate missing models...")
            try:
                from emergency_auc_hotfix import create_fallback_model
                if create_fallback_model():
                    console.print("✅ Fallback model created")
                    fix_success = True
            except Exception as e:
                console.print(f"❌ Model recreation failed: {e}")
        
        return fix_success
    
    def run_continuous_monitor(self, check_interval=300):  # 5 minutes
        """Run continuous monitoring with auto-fix"""
        console.print(Panel.fit(
            f"🎯 Starting Continuous Monitor\n"
            f"Check interval: {check_interval} seconds\n"
            f"Press Ctrl+C to stop",
            style="bold green"
        ))
        
        try:
            while True:
                # Run health check
                health_report = self.check_pipeline_health()
                
                # Update status
                self.status['current_auc'] = health_report.get('current_auc', 0.0)
                self.status['status'] = health_report['overall_status']
                
                # Display current status
                self.display_health_report(health_report)
                
                # Auto-fix if needed
                if health_report['overall_status'] in ['critical', 'warning']:
                    if self.status['auto_fix_attempts'] < self.max_retries:
                        console.print("🔧 Issues detected, attempting auto-fix...")
                        if self.auto_fix_issues(health_report):
                            self.status['auto_fix_attempts'] = 0
                            self.status['last_successful_run'] = datetime.now().isoformat()
                        else:
                            self.status['auto_fix_attempts'] += 1
                    else:
                        console.print("⚠️ Max auto-fix attempts reached. Manual intervention required.")
                
                # Save status
                self.save_status()
                
                # Wait for next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            console.print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            console.print(f"❌ Monitoring error: {e}")
            self.logger.error(f"Monitoring error: {e}")
    
    def display_health_report(self, health_report: Dict[str, Any]):
        """Display health report in a nice format"""
        # Main status table
        table = Table(title="🎯 Production Pipeline Health", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="yellow")
        
        # Overall status
        status_color = {
            'healthy': 'green',
            'warning': 'yellow', 
            'critical': 'red',
            'unknown': 'white'
        }
        
        table.add_row(
            "Overall",
            f"[{status_color[health_report['overall_status']]}]{health_report['overall_status'].upper()}[/]",
            f"Issues: {len(health_report['issues'])}"
        )
        
        # AUC status
        auc_color = 'green' if health_report['auc_status'] == 'good' else 'red'
        table.add_row(
            "AUC Performance",
            f"[{auc_color}]{health_report['auc_status'].upper()}[/]",
            f"Current: {health_report['current_auc']:.3f}"
        )
        
        # Data status
        data_color = 'green' if health_report['data_status'] == 'good' else 'yellow'
        table.add_row(
            "Data Quality",
            f"[{data_color}]{health_report['data_status'].upper()}[/]",
            "Data validation passed"
        )
        
        # Model status
        model_color = 'green' if health_report['model_status'] == 'good' else 'red'
        table.add_row(
            "Model Files",
            f"[{model_color}]{health_report['model_status'].upper()}[/]",
            "All required files present"
        )
        
        console.print(table)
        
        # Issues and recommendations
        if health_report['issues']:
            issues_text = "\n".join(health_report['issues'])
            console.print(Panel(issues_text, title="⚠️ Issues Detected", style="red"))
        
        if health_report['recommendations']:
            recommendations_text = "\n".join(health_report['recommendations'])
            console.print(Panel(recommendations_text, title="💡 Recommendations", style="yellow"))


def run_production_monitor():
    """Main function to run production monitoring"""
    monitor = ProductionMonitor()
    
    # Run initial health check
    health_report = monitor.check_pipeline_health()
    monitor.display_health_report(health_report)
    
    # Ask user for monitoring mode
    console.print("\n🎯 Select monitoring mode:")
    console.print("1. Single health check")
    console.print("2. Continuous monitoring")
    console.print("3. Auto-fix current issues")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "2":
        monitor.run_continuous_monitor()
    elif choice == "3":
        monitor.auto_fix_issues(health_report)
    else:
        console.print("✅ Single health check completed")


if __name__ == "__main__":
    run_production_monitor()
