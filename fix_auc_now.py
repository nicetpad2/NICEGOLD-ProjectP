#!/usr/bin/env python3
"""
🚀 INSTANT AUC FIX LAUNCHER
===========================
แก้ไขปัญหา AUC ต่ำทันทีด้วยระบบอัตโนมัติ

Usage:
    python fix_auc_now.py                    # Run quick fix
    python fix_auc_now.py --full             # Run complete fix
    python fix_auc_now.py --emergency        # Emergency hotfix only
    python fix_auc_now.py --monitor          # Start monitoring
"""

import sys
import argparse
import os
from pathlib import Path

# Rich console for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

console = Console()

def run_emergency_fix():
    """Run emergency AUC hotfix"""
    console.print(Panel.fit("🚨 EMERGENCY AUC HOTFIX", style="bold red"))
    
    try:
        from emergency_auc_hotfix import emergency_auc_hotfix
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running emergency hotfix...", total=None)
            success = emergency_auc_hotfix()
            progress.update(task, description="Emergency hotfix completed")
        
        if success:
            console.print("✅ Emergency hotfix successful!")
            return True
        else:
            console.print("⚠️ Emergency hotfix completed with warnings")
            return False
            
    except Exception as e:
        console.print(f"❌ Emergency hotfix failed: {e}")
        return False

def run_full_fix():
    """Run complete production AUC fix"""
    console.print(Panel.fit("🚀 PRODUCTION AUC COMPLETE FIX", style="bold blue"))
    
    try:
        from production_auc_critical_fix import run_production_auc_fix
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running complete AUC fix...", total=None)
            results = run_production_auc_fix()
            progress.update(task, description="Complete fix finished")
        
        if results['success']:
            console.print(f"✅ Complete fix successful! AUC: {results['final_auc']:.4f}")
            return True
        else:
            console.print(f"⚠️ Fix completed but needs more work. AUC: {results['final_auc']:.4f}")
            return False
            
    except Exception as e:
        console.print(f"❌ Complete fix failed: {e}")
        return False

def run_quick_fix():
    """Run quick AUC fix (tries emergency first, then full if needed)"""
    console.print(Panel.fit("⚡ QUICK AUC FIX", style="bold green"))
    
    # Try emergency fix first
    console.print("🔧 Step 1: Attempting emergency fix...")
    if run_emergency_fix():
        console.print("✅ Emergency fix solved the issue!")
        return True
    
    # If emergency fix didn't work, try full fix
    console.print("🔧 Step 2: Running complete fix...")
    if run_full_fix():
        console.print("✅ Complete fix solved the issue!")
        return True
    
    console.print("⚠️ Both fixes completed but may need manual intervention")
    return False

def start_monitoring():
    """Start production monitoring"""
    console.print(Panel.fit("🎯 PRODUCTION MONITORING", style="bold purple"))
    
    try:
        from production_monitor import run_production_monitor
        run_production_monitor()
    except Exception as e:
        console.print(f"❌ Monitoring failed: {e}")

def check_current_status():
    """Check current AUC status"""
    console.print(Panel.fit("🔍 CURRENT STATUS CHECK", style="bold cyan"))
    
    try:
        # Check if prediction files exist
        output_dir = Path("output_default")
        pred_file = output_dir / "predictions.csv"
        metrics_file = output_dir / "predict_summary_metrics.json"
        
        table = Table(title="📊 Current Status", box=box.ROUNDED)
        table.add_column("Item", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="yellow")
        
        # Check prediction file
        if pred_file.exists():
            table.add_row("Predictions", "✅ EXISTS", str(pred_file))
        else:
            table.add_row("Predictions", "❌ MISSING", "No predictions found")
        
        # Check AUC
        if metrics_file.exists():
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                auc = metrics.get('auc', 0.0)
                
            if auc >= 0.70:
                status = "✅ GOOD"
                style = "green"
            elif auc >= 0.60:
                status = "⚠️ WARNING"
                style = "yellow"
            else:
                status = "❌ CRITICAL"
                style = "red"
                
            table.add_row("AUC Score", f"[{style}]{status}[/]", f"{auc:.4f}")
        else:
            table.add_row("AUC Score", "❓ UNKNOWN", "No metrics file")
        
        # Check model files
        model_file = output_dir / "catboost_model_best_cv.pkl"
        features_file = output_dir / "train_features.txt"
        
        if model_file.exists() and features_file.exists():
            table.add_row("Model Files", "✅ READY", "All files present")
        else:
            table.add_row("Model Files", "❌ INCOMPLETE", "Missing model files")
        
        console.print(table)
        
        # Return AUC for decision making
        if metrics_file.exists():
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                return metrics.get('auc', 0.0)
        
        return 0.0
        
    except Exception as e:
        console.print(f"❌ Status check failed: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="🚀 Instant AUC Fix Launcher")
    parser.add_argument("--emergency", action="store_true", help="Run emergency hotfix only")
    parser.add_argument("--full", action="store_true", help="Run complete production fix")
    parser.add_argument("--monitor", action="store_true", help="Start production monitoring")
    parser.add_argument("--status", action="store_true", help="Check current status only")
    
    args = parser.parse_args()
    
    # Show welcome message
    console.print(Panel.fit(
        "🚀 INSTANT AUC FIX LAUNCHER\n"
        "Automated system to fix AUC performance issues\n"
        "Target: AUC ≥ 0.70 for production readiness",
        style="bold white on blue"
    ))
    
    # Check current status first
    current_auc = check_current_status()
    
    # Route to appropriate action
    if args.status:
        return
    elif args.monitor:
        start_monitoring()
    elif args.emergency:
        run_emergency_fix()
    elif args.full:
        run_full_fix()
    else:
        # Intelligent routing based on current status
        if current_auc >= 0.70:
            console.print("✅ AUC is already good! No fix needed.")
        elif current_auc >= 0.60:
            console.print("⚠️ AUC needs improvement. Running quick fix...")
            run_quick_fix()
        else:
            console.print("🚨 Critical AUC issue. Running comprehensive fix...")
            run_quick_fix()
    
    # Final status check
    console.print("\n" + "="*50)
    final_auc = check_current_status()
    
    if final_auc >= 0.70:
        console.print(Panel.fit(
            f"🎉 SUCCESS! AUC is now {final_auc:.4f}\n"
            "✅ System is ready for production!",
            style="bold green"
        ))
    elif final_auc >= 0.60:
        console.print(Panel.fit(
            f"⚠️ PARTIAL SUCCESS. AUC is now {final_auc:.4f}\n"
            "Consider running --full for better results",
            style="bold yellow"
        ))
    else:
        console.print(Panel.fit(
            f"❌ NEEDS MORE WORK. AUC is {final_auc:.4f}\n"
            "Manual intervention may be required",
            style="bold red"
        ))

if __name__ == "__main__":
    main()
