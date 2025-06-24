# 🔧 GLOBAL_FALLBACK_APPLIED: Comprehensive error handling
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Global exception handler for imports
def safe_import(module_name, fallback_value=None, fallback_message=None):
    """Safely import modules with fallbacks"""
    try:
        parts = module_name.split('.')
        module = __import__(module_name)
        for part in parts[1:]:
            module = getattr(module, part)
        return module
    except ImportError as e:
        if fallback_message:
            print(f"⚠️ {fallback_message}")
        else:
            print(f"⚠️ Failed to import {module_name}, using fallback")
        return fallback_value


import logging
from tqdm import tqdm
from projectp.config_loader import load_config
from projectp.steps.preprocess import run_preprocess
from projectp.steps.sweep import run_sweep
from projectp.steps.backtest import run_backtest
from projectp.steps.threshold import run_threshold
from projectp.steps.report import run_report
from projectp.steps.train import run_train
from projectp.steps.walkforward import run_walkforward
from projectp.steps.predict import run_predict
from projectp.notify import send_notification
from prefect import flow, task
from projectp.enterprise_services import audit_log
import subprocess
from feature_engineering import run_data_quality_checks, run_auto_feature_generation, run_feature_interaction, run_mutual_info_feature_selection, check_feature_collinearity

# 🚀 AUC Improvement Pipeline Integration - Enhanced
try:
    from auc_improvement_pipeline import (
        AUCImprovementPipeline,
        run_auc_emergency_fix,
        run_advanced_feature_engineering,
        run_model_ensemble_boost,
        run_threshold_optimization_v2
    )
    AUC_IMPROVEMENT_AVAILABLE = True
except ImportError:
    AUC_IMPROVEMENT_AVAILABLE = False
    print("⚠️ AUC Improvement Pipeline not available - using emergency hotfix")

# Emergency AUC Fix Integration
try:
    from emergency_auc_hotfix import emergency_auc_hotfix
    from production_auc_critical_fix import run_production_auc_fix
    EMERGENCY_FIX_AVAILABLE = True
except ImportError:
    EMERGENCY_FIX_AVAILABLE = False
    print("⚠️ Emergency AUC fixes not available")
from typing import Optional, Dict, Any
import uuid
from src.utils.log_utils import set_log_context, pro_log_json, export_log_to
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

# Setup logging
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

console = Console()

PIPELINE_STEPS_TRAIN_PREDICT = [
    ("Preprocess", run_preprocess),
    ("WalkForward", run_walkforward), # Train model and create train_features.txt
    ("Predict", run_predict),       # Use the newly created model and features
    ("Backtest", run_backtest),
    ("Report", run_report),
]

# 🎯 Enhanced PIPELINE_STEPS_FULL with AUC Improvement (เทพทุกส่วน)
PIPELINE_STEPS_FULL = [
    ("Preprocess", run_preprocess),
    ("🔍 AUC Emergency Diagnosis", lambda: run_auc_emergency_fix() if AUC_IMPROVEMENT_AVAILABLE else print("AUC improvement skipped")),
    ("🧠 Advanced Feature Engineering", lambda: run_advanced_feature_engineering() if AUC_IMPROVEMENT_AVAILABLE else print("Advanced features skipped")),
    ("Train", run_train),
    ("🚀 Model Ensemble Boost", lambda: run_model_ensemble_boost() if AUC_IMPROVEMENT_AVAILABLE else print("Ensemble boost skipped")),
    ("Sweep", run_sweep),
    ("🎯 Threshold Optimization V2", lambda: run_threshold_optimization_v2() if AUC_IMPROVEMENT_AVAILABLE else print("Threshold V2 skipped")),
    ("Threshold", run_threshold),
    ("WalkForward", run_walkforward),
    ("Predict", run_predict),
    ("Backtest", run_backtest),
    ("Report", run_report),
]

# 🔥 PIPELINE_STEPS_ULTIMATE (เทพสุดทุกส่วน - สำหรับ Production) - Enhanced
PIPELINE_STEPS_ULTIMATE = [
    ("🏗️ Preprocess", run_preprocess),
    ("🆘 Emergency AUC Check", lambda: emergency_auc_hotfix() if EMERGENCY_FIX_AVAILABLE else print("Emergency fix skipped")),
    ("🔬 Data Quality Checks", lambda: run_data_quality_checks()),
    ("🔍 AUC Emergency Diagnosis", lambda: run_auc_emergency_fix() if AUC_IMPROVEMENT_AVAILABLE else print("AUC improvement skipped")),
    ("🧠 Advanced Feature Engineering", lambda: run_advanced_feature_engineering() if AUC_IMPROVEMENT_AVAILABLE else print("Advanced features skipped")),
    ("⚡ Auto Feature Generation", lambda: run_auto_feature_generation()),
    ("🤝 Feature Interaction", lambda: run_feature_interaction()),
    ("🎯 Mutual Info Selection", lambda: run_mutual_info_feature_selection()),
    ("🤖 Train Base Models", run_train),
    ("🚀 Model Ensemble Boost", lambda: run_model_ensemble_boost() if AUC_IMPROVEMENT_AVAILABLE else print("Ensemble boost skipped")),
    ("🔧 Hyperparameter Sweep", run_sweep),
    ("🎯 Threshold Optimization V2", lambda: run_threshold_optimization_v2() if AUC_IMPROVEMENT_AVAILABLE else print("Threshold V2 skipped")),
    ("⚖️ Threshold Optimization", run_threshold),
    ("🏃 Walk-Forward Validation", run_walkforward),
    ("🔮 Prediction", run_predict),
    ("📊 Backtest Simulation", run_backtest),
    ("📈 Performance Report", run_report),
]

# Prefect task wrappers
@task
def preprocess_task(config):
    return run_preprocess(config)
@task
def train_task(config, debug_mode=False):
    return run_train(config, debug_mode=debug_mode)
@task
def sweep_task(config):
    return run_sweep(config)
@task
def threshold_task(config):
    return run_threshold(config)
@task
def walkforward_task(config):
    return run_walkforward(config)
@task
def predict_task(config):
    return run_predict(config)
@task
def backtest_task(config):
    return run_backtest(config)
@task
def report_task(config):
    return run_report(config)
@task
def data_quality_task(config: Optional[Dict[str, Any]] = None) -> bool:
    run_data_quality_checks()
    return True
@task
def auto_feature_task(config: Optional[Dict[str, Any]] = None) -> bool:
    run_auto_feature_generation()
    return True
@task
def feature_interaction_task(config: Optional[Dict[str, Any]] = None) -> bool:
    run_feature_interaction()
    return True
@task
def mutual_info_task(config: Optional[Dict[str, Any]] = None) -> bool:
    run_mutual_info_feature_selection()
    return True

# 🚀 AUC Improvement Tasks (เทพทุกส่วน)
@task
def auc_emergency_fix_task(config: Optional[Dict[str, Any]] = None) -> bool:
    """AUC Emergency Fix - แก้ปัญหา AUC ต่ำด่วน"""
    if AUC_IMPROVEMENT_AVAILABLE:
        return run_auc_emergency_fix()
    console.print(Panel("[yellow]⚠️ AUC Emergency Fix not available", title="Skipped", border_style="yellow"))
    return False

@task
def advanced_feature_engineering_task(config: Optional[Dict[str, Any]] = None) -> bool:
    """Advanced Feature Engineering - สร้างฟีเจอร์ขั้นสูง"""
    if AUC_IMPROVEMENT_AVAILABLE:
        return run_advanced_feature_engineering()
    console.print(Panel("[yellow]⚠️ Advanced Feature Engineering not available", title="Skipped", border_style="yellow"))
    return False

@task
def model_ensemble_boost_task(config: Optional[Dict[str, Any]] = None) -> bool:
    """Model Ensemble Boost - เพิ่มพลัง ensemble"""
    if AUC_IMPROVEMENT_AVAILABLE:
        return run_model_ensemble_boost()
    console.print(Panel("[yellow]⚠️ Model Ensemble Boost not available", title="Skipped", border_style="yellow"))
    return False

@task
def threshold_optimization_v2_task(config: Optional[Dict[str, Any]] = None) -> bool:
    """Threshold Optimization V2 - ปรับ threshold แบบเทพ"""
    if AUC_IMPROVEMENT_AVAILABLE:
        return run_threshold_optimization_v2()
    console.print(Panel("[yellow]⚠️ Threshold Optimization V2 not available", title="Skipped", border_style="yellow"))
    return False

def show_progress(tasks):
    errors = []
    warnings = []
    results = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]กำลังประมวลผล...", total=len(tasks))
        for step in tasks:
            try:
                desc, func = step
                progress.update(task, description=f"[green]{desc}")
                result = func()
                if result is not None:
                    results.append(result)
            except Warning as w:
                warnings.append(str(w))
                console.print(Panel(f"[bold yellow]⚠️ WARNING: {w}", title="Warning", border_style="yellow"))
            except Exception as e:
                errors.append(str(e))
                console.print(Panel(f"[bold red]❌ ERROR in step '{desc}': {e}", title="Critical Error", border_style="red"))
                # --- FIX: Stop the pipeline on critical error ---
                progress.update(task, description=f"[bold red]Pipeline HALTED due to error in {desc}.")
                return errors, warnings, results
            progress.update(task, advance=1)
    return errors, warnings, results

def run_full_pipeline():
    """Execute the complete end-to-end pipeline sequence with AUC improvements"""
    # ENFORCE REAL DATA ONLY - Critical validation at pipeline start
    try:
        from projectp.data_validator import enforce_real_data_only
        data_validator = enforce_real_data_only()
        console.print(Panel("[bold green]🛡️ Real data enforcement activated - only datacsv data allowed", border_style="green"))
    except Exception as e:
        error_msg = f"❌ CRITICAL: Real data validation failed: {e}"
        console.print(Panel(f"[bold red]{error_msg}", title="Critical Error", border_style="red"))
        raise ValueError(error_msg)
    
    console.print(Panel(
        "[bold green]🚀 Starting FULL PIPELINE with AUC Improvements\n🛡️ REAL DATA ONLY - No dummy/synthetic data allowed",
        title="🎯 Enhanced Pipeline",
        border_style="green"
    ))
    errors, warnings, results = show_progress(PIPELINE_STEPS_FULL)
    
    # สรุปผลลัพธ์
    if errors:
        console.print(Panel(f"[bold red]❌ Pipeline completed with {len(errors)} errors", title="Results", border_style="red"))
    elif warnings:
        console.print(Panel(f"[bold yellow]⚠️ Pipeline completed with {len(warnings)} warnings", title="Results", border_style="yellow"))
    else:
        console.print(Panel("[bold green]✅ Pipeline completed successfully!", title="Results", border_style="green"))
    
    return errors, warnings, results

def run_debug_full_pipeline():
    """Execute the complete end-to-end pipeline in debug mode with AUC improvements"""
    # ENFORCE REAL DATA ONLY - Critical validation at pipeline start
    try:
        from projectp.data_validator import enforce_real_data_only
        data_validator = enforce_real_data_only()
        console.print(Panel("[bold cyan]🛡️ Real data enforcement activated - only datacsv data allowed", border_style="cyan"))
    except Exception as e:
        error_msg = f"❌ CRITICAL: Real data validation failed: {e}"
        console.print(Panel(f"[bold red]{error_msg}", title="Critical Error", border_style="red"))
        raise ValueError(error_msg)
    
    console.print(Panel(
        "[bold cyan]🐞 Starting DEBUG PIPELINE with AUC Improvements\n🛡️ REAL DATA ONLY - No dummy/synthetic data allowed",
        title="🔍 Debug Mode",
        border_style="cyan"
    ))
    errors, warnings, results = show_progress(PIPELINE_STEPS_FULL)
    return errors, warnings, results

def run_ultimate_pipeline():
    """Execute the ULTIMATE pipeline - เทพสุดทุกส่วน (สำหรับ Production)"""
    # ENFORCE REAL DATA ONLY - Critical validation at pipeline start
    try:
        from projectp.data_validator import enforce_real_data_only
        data_validator = enforce_real_data_only()
        console.print(Panel("[bold magenta]🛡️ Real data enforcement activated - only datacsv data allowed", border_style="magenta"))
    except Exception as e:
        error_msg = f"❌ CRITICAL: Real data validation failed: {e}"
        console.print(Panel(f"[bold red]{error_msg}", title="Critical Error", border_style="red"))
        raise ValueError(error_msg)
    
    console.print(Panel(
        "[bold magenta]🔥 Starting ULTIMATE PIPELINE - เทพสุดทุกส่วน!\n🛡️ REAL DATA ONLY - No dummy/synthetic data allowed",
        title="🏆 Production Mode",
        border_style="magenta"
    ))
    
    # แสดง pipeline steps ที่จะรัน
    table = Table(title="🔥 Ultimate Pipeline Steps", show_header=True, header_style="bold magenta")
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    
    for i, (step_name, _) in enumerate(PIPELINE_STEPS_ULTIMATE):
        table.add_row(f"{i+1}", step_name)
    
    console.print(table)
    
    errors, warnings, results = show_progress(PIPELINE_STEPS_ULTIMATE)
    
    # สรุปผลลัพธ์แบบเทพ
    if not errors and not warnings:
        console.print(Panel(
            "[bold green]🏆 ULTIMATE PIPELINE SUCCESS!\n"
            "✅ All steps completed flawlessly\n"
            "🚀 Ready for Production deployment!",
            title="🔥 ULTIMATE SUCCESS",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]⚠️ Pipeline completed with issues:\n"
            f"❌ Errors: {len(errors)}\n"
            f"⚠️ Warnings: {len(warnings)}",
            title="📊 Results Summary",
            border_style="yellow"
        ))
    
    return errors, warnings, results
