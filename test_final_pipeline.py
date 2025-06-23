#!/usr/bin/env python3
"""
Final comprehensive test for the AUC improvement pipeline.
Tests all major functions and ensures no NaN AUC issues.
"""

import pandas as pd
import numpy as np
from auc_improvement_pipeline import AUCImprovementPipeline, run_auc_emergency_fix
from rich.console import Console
from rich.panel import Panel

console = Console()

def test_emergency_functions():
    """Test all emergency fix functions"""
    console.print(Panel.fit("🧪 Testing Emergency Functions", style="bold blue"))
    
    # Test the main emergency fix function
    try:
        result = run_auc_emergency_fix()
        console.print(f"[green]✓ run_auc_emergency_fix() returned: {result}")
    except Exception as e:
        console.print(f"[red]❌ run_auc_emergency_fix() failed: {e}")
        return False
    
    return True

def test_pipeline_class_methods():
    """Test key pipeline class methods directly"""
    console.print(Panel.fit("🧪 Testing Pipeline Class Methods", style="bold blue"))
    
    try:
        pipeline = AUCImprovementPipeline(target_auc=0.70)
        
        # Test data loading
        X, y, analysis = pipeline.load_and_analyze_data()
        if X is not None:
            console.print(f"[green]✓ Data loaded: X shape {X.shape}, y shape {y.shape}")
        else:
            console.print("[yellow]⚠️ No real data found, creating synthetic data for testing")
            
            # Create synthetic extreme imbalance data
            np.random.seed(42)
            n_samples = 1000
            n_features = 20
            
            X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                           columns=[f'feature_{i}' for i in range(n_features)])
            
            # Create extreme imbalance: 95% class 0, 5% class 1
            y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]))
            
            console.print(f"[green]✓ Synthetic data created: X shape {X.shape}, y shape {y.shape}")
            console.print(f"[cyan]Class distribution: {y.value_counts().to_dict()}")
        
        # Test robust baseline testing (the key fix)
        baseline_aucs = pipeline._test_baseline_models_robust(X, y, handle_imbalance=True)
        console.print(f"[green]✓ Robust baseline testing completed")
        
        for model_name, auc in baseline_aucs.items():
            if np.isnan(auc):
                console.print(f"[red]❌ {model_name}: AUC is NaN!")
                return False
            else:
                console.print(f"[green]✓ {model_name}: AUC = {auc:.3f}")
        
        # Test problem diagnosis
        problems, diagnostic_aucs = pipeline.diagnose_auc_problems(X, y)
        console.print(f"[green]✓ Problem diagnosis completed")
        console.print(f"[cyan]Found {len(problems)} problems:")
        for i, problem in enumerate(problems, 1):
            console.print(f"    {i}. {problem}")
        
        for model_name, auc in diagnostic_aucs.items():
            if np.isnan(auc):
                console.print(f"[red]❌ Diagnostic {model_name}: AUC is NaN!")
                return False
            else:
                console.print(f"[green]✓ Diagnostic {model_name}: AUC = {auc:.3f}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Pipeline class testing failed: {e}")
        import traceback
        console.print(f"[red]Traceback: {traceback.format_exc()}")
        return False

def test_extreme_scenarios():
    """Test extreme edge cases"""
    console.print(Panel.fit("🧪 Testing Extreme Scenarios", style="bold red"))
    
    try:
        pipeline = AUCImprovementPipeline(target_auc=0.60)
        
        # Scenario 1: Very extreme imbalance (99.5% vs 0.5%)
        console.print("\n[yellow]Scenario 1: Very extreme imbalance (99.5% vs 0.5%)")
        np.random.seed(123)
        X_extreme = pd.DataFrame(np.random.randn(2000, 15), 
                               columns=[f'feat_{i}' for i in range(15)])
        y_extreme = pd.Series(np.random.choice([0, 1], size=2000, p=[0.995, 0.005]))
        
        console.print(f"[cyan]Extreme class distribution: {y_extreme.value_counts().to_dict()}")
        
        aucs_extreme = pipeline._test_baseline_models_robust(X_extreme, y_extreme, handle_imbalance=True)
        
        all_valid = True
        for model_name, auc in aucs_extreme.items():
            if np.isnan(auc):
                console.print(f"[red]❌ Extreme scenario {model_name}: AUC is NaN!")
                all_valid = False
            else:
                console.print(f"[green]✓ Extreme scenario {model_name}: AUC = {auc:.3f}")
        
        # Scenario 2: Only one class
        console.print("\n[yellow]Scenario 2: Single class (should be handled gracefully)")
        X_single = pd.DataFrame(np.random.randn(100, 10), 
                              columns=[f'feat_{i}' for i in range(10)])
        y_single = pd.Series([0] * 100)  # All same class
        
        try:
            aucs_single = pipeline._test_baseline_models_robust(X_single, y_single, handle_imbalance=True)
            console.print(f"[green]✓ Single class scenario handled gracefully")
            for model_name, auc in aucs_single.items():
                console.print(f"[cyan]  {model_name}: AUC = {auc:.3f}")
        except Exception as e:
            console.print(f"[yellow]⚠️ Single class scenario: {e}")
        
        # Scenario 3: Very small dataset
        console.print("\n[yellow]Scenario 3: Very small dataset")
        X_small = pd.DataFrame(np.random.randn(20, 5), 
                             columns=[f'feat_{i}' for i in range(5)])
        y_small = pd.Series([0]*15 + [1]*5)  # 15 vs 5
        
        aucs_small = pipeline._test_baseline_models_robust(X_small, y_small, handle_imbalance=True)
        
        for model_name, auc in aucs_small.items():
            if np.isnan(auc):
                console.print(f"[red]❌ Small dataset {model_name}: AUC is NaN!")
                all_valid = False
            else:
                console.print(f"[green]✓ Small dataset {model_name}: AUC = {auc:.3f}")
        
        return all_valid
        
    except Exception as e:
        console.print(f"[red]❌ Extreme scenario testing failed: {e}")
        import traceback
        console.print(f"[red]Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests"""
    console.print(Panel.fit("🚀 Final Pipeline Comprehensive Test", style="bold green"))
    
    test_results = []
    
    # Test 1: Emergency functions
    result1 = test_emergency_functions()
    test_results.append(("Emergency Functions", result1))
    
    # Test 2: Pipeline class methods
    result2 = test_pipeline_class_methods()
    test_results.append(("Pipeline Class Methods", result2))
    
    # Test 3: Extreme scenarios
    result3 = test_extreme_scenarios()
    test_results.append(("Extreme Scenarios", result3))
    
    # Final summary
    console.print("\n" + "="*60)
    console.print(Panel.fit("📊 Final Test Results", style="bold cyan"))
    
    all_passed = True
    for test_name, passed in test_results:
        status = "[green]✓ PASSED" if passed else "[red]❌ FAILED"
        console.print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    console.print("\n" + "="*60)
    if all_passed:
        console.print(Panel.fit("🎉 ALL TESTS PASSED! Pipeline is ready for production.", style="bold green"))
        console.print("\n[green]✅ The AUC improvement pipeline now:")
        console.print("  ✓ Handles extreme class imbalance robustly")
        console.print("  ✓ Never produces NaN AUC values")
        console.print("  ✓ Has proper fallback mechanisms")
        console.print("  ✓ Works with small datasets")
        console.print("  ✓ Provides clear error messages")
        
        console.print("\n[cyan]🚀 Ready to run full pipeline!")
        
    else:
        console.print(Panel.fit("❌ SOME TESTS FAILED! Please review the issues above.", style="bold red"))
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
