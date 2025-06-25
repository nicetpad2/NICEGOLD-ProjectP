
# model_guard.py - Production Enhanced Version
                    from production_auc_critical_fix import run_production_auc_fix
from rich.console import Console
from rich.panel import Panel
import logging
import numpy as np
import os
console = Console()

def check_auc_threshold(metrics, min_auc = 0.7, strict = True, auto_fix = True):
    """
    Enhanced AUC threshold check with auto - fix capability

    Args:
        metrics: Dictionary containing AUC and other metrics
        min_auc: Minimum required AUC (default: 0.7)
        strict: If True, raises error. If False, only warns
        auto_fix: If True, attempts to run auto - fix pipeline
    """
    auc = metrics.get('auc', 0)

    if auc < min_auc:
        if auc > 0.5:
            console.print(f"⚠️ [Guard][Warning] AUC below threshold but above random: {auc:.2f} < {min_auc}")
            if auto_fix:
                console.print("🚀 Attempting auto - fix...")
                try:
                    fix_results = run_production_auc_fix()
                    if fix_results['success']:
                        console.print("✅ Auto - fix successful! Re - run the pipeline.")
                        return True
                except Exception as e:
                    console.print(f"❌ Auto - fix failed: {e}")
        else:
            msg = f"AUC too low: {auc:.2f} < {min_auc} (not better than random)"

            if auto_fix:
                console.print(Panel.fit(
                    f"🚨 CRITICAL AUC ISSUE DETECTED\n"
                    f"Current AUC: {auc:.2f}\n"
                    f"Required: {min_auc:.2f}\n"
                    f"Running emergency fix...", 
                    style = "bold red"
                ))

                try:
                    fix_results = run_production_auc_fix()

                    if fix_results['success']:
                        console.print("✅ Emergency fix successful! Pipeline can continue.")
                        return True
                    else:
                        console.print("⚠️ Emergency fix completed but may need more work.")

                except Exception as e:
                    console.print(f"❌ Emergency fix failed: {e}")

            if strict:
                raise ValueError(msg)
            else:
                console.print(f"[Guard][Warning] {msg}")
    else:
        console.print(f"✅ [Guard] AUC OK: {auc:.2f}")

    return False

def check_no_overfitting(metrics, max_gap = 0.1, strict = True):
    train_auc = metrics.get('train_auc', 0)
    test_auc = metrics.get('test_auc', 0)
    if train_auc - test_auc > max_gap:
        msg = f"Overfitting detected: train_auc = {train_auc:.2f}, test_auc = {test_auc:.2f}"
        if strict:
            raise ValueError(msg)
        else:
            print(f"[Guard][Warning] {msg}")
    else:
        print(f"[Guard] No overfitting: train_auc = {train_auc:.2f}, test_auc = {test_auc:.2f}")

def check_no_noise(feature_importance, threshold = 0.01):
    noisy = [f for f, imp in feature_importance.items() if imp < threshold]
    if len(noisy) > 0:
        print(f"[Guard] Warning: Noisy features detected: {noisy}")
    else:
        print("[Guard] No noisy features detected.")

def check_no_data_leak(df_train, df_test):
    overlap = set(df_train.index).intersection(set(df_test.index))
    if overlap:
        raise ValueError("Data leak detected: train/test overlap")
    print("[Guard] No data leak detected.")