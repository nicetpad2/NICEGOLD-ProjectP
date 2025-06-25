from projectp.steps.train import (
import asyncio
"""
Training Module Usage Examples
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
Comprehensive examples for using the training module
"""

    run_train, 
    AdvancedTrainer, 
    run_ensemble_train, 
    run_hyperopt_train, 
    run_cv_train, 
    run_automl_train, 
    PerformanceMonitor, 
    TrainingDashboard, 
    TrainingReporter
)

def example_basic_training():
    """Example: Basic training pipeline"""
    print("🚀 Example 1: Basic Training Pipeline")

    config = {
        'model_type': 'catboost', 
        'target_auc': 70.0, 
        'max_iterations': 5, 
        'verbose': True
    }

    results = run_train(config, target_auc = 70.0, max_iterations = 5)

    print(f"Training completed: {results.get('success', False)}")
    print(f"Best AUC: {results.get('best_auc', 0):.2f}%")
    print(f"Target achieved: {results.get('target_achieved', False)}")

    return results

def example_advanced_training():
    """Example: Advanced training with monitoring"""
    print("\n🔧 Example 2: Advanced Training with Monitoring")

    # Setup performance monitoring
    monitor = PerformanceMonitor(monitoring_interval = 0.5)

    # Setup training configuration
    config = {
        'model_type': 'catboost', 
        'target_auc': 75.0, 
        'max_iterations': 3
    }

    # Create advanced trainer
    trainer = AdvancedTrainer(config)

    try:
        # Start monitoring
        monitor.start_monitoring()

        # Run training
        results = trainer.run_full_pipeline(target_auc = 75.0, max_iterations = 3)

        # Update monitor with final results
        if results.get('success', False):
            monitor.update_training_progress(
                iteration = results.get('best_iteration', 0), 
                auc = results.get('best_auc', 0.0)
            )

        print(f"Advanced training completed: {results.get('success', False)}")
        print(f"Best AUC: {results.get('best_auc', 0):.2f}%")

        return results

    finally:
        # Stop monitoring and generate report
        monitor.stop_monitoring()

        # Generate performance report
        reporter = TrainingReporter(monitor)
        reporter.display_final_report()

async def example_ensemble_training():
    """Example: Ensemble training"""
    print("\n🎯 Example 3: Ensemble Training")

    config = {
        'target_auc': 70.0, 
        'verbose': True
    }

    results = await run_ensemble_train(
        config = config, 
        target_auc = 70.0, 
        ensemble_size = 3
    )

    print(f"Ensemble training completed: {results.get('success', False)}")
    print(f"Ensemble AUC: {results.get('ensemble_auc', 0):.2f}%")
    print(f"Models trained: {results.get('models_trained', 0)}")

    return results

def example_hyperparameter_optimization():
    """Example: Hyperparameter optimization"""
    print("\n🔧 Example 4: Hyperparameter Optimization")

    config = {
        'target_auc': 72.0, 
        'verbose': True
    }

    results = run_hyperopt_train(
        config = config, 
        target_auc = 72.0, 
        n_trials = 10  # Reduced for demo
    )

    print(f"Optimization completed: {results.get('success', False)}")
    print(f"Best AUC: {results.get('best_auc', 0):.2f}%")
    print(f"Best parameters: {results.get('best_params', {})}")

    return results

def example_cross_validation():
    """Example: Cross - validation training"""
    print("\n📊 Example 5: Cross - Validation Training")

    config = {
        'model_type': 'catboost', 
        'target_auc': 68.0, 
        'verbose': True
    }

    results = run_cv_train(
        config = config, 
        target_auc = 68.0, 
        cv_folds = 3  # Reduced for demo
    )

    print(f"CV training completed: {results.get('success', False)}")
    print(f"CV Mean AUC: {results.get('cv_mean_auc', 0):.2f}%")
    print(f"CV Std AUC: {results.get('cv_std_auc', 0):.2f}%")

    return results

def example_automated_model_selection():
    """Example: Automated model selection"""
    print("\n🤖 Example 6: Automated Model Selection")

    config = {
        'target_auc': 69.0, 
        'verbose': True
    }

    results = run_automl_train(
        config = config, 
        target_auc = 69.0
    )

    print(f"AutoML completed: {results.get('success', False)}")
    print(f"Best model: {results.get('best_model', {}).get('name', 'N/A')}")
    print(f"Best AUC: {results.get('best_auc', 0):.2f}%")

    return results

def example_live_dashboard():
    """Example: Live training dashboard"""
    print("\n📊 Example 7: Live Training Dashboard")

    # Setup monitoring
    monitor = PerformanceMonitor(monitoring_interval = 0.5)
    dashboard = TrainingDashboard(monitor)

    config = {
        'model_type': 'catboost', 
        'target_auc': 70.0, 
        'max_iterations': 3
    }

    try:
        # Start monitoring
        monitor.start_monitoring()

        # Start dashboard in background (for demo, we'll skip the live part)
        print("🖥️ Dashboard would start here (live display)")

        # Run training
        trainer = AdvancedTrainer(config)
        results = trainer.run_full_pipeline(target_auc = 70.0, max_iterations = 3)

        print(f"Training with dashboard completed: {results.get('success', False)}")

        return results

    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        dashboard.stop_dashboard()

def run_all_examples():
    """Run all training examples"""
    print(" = " * 80)
    print("🎓 ML Training Module - Complete Examples")
    print(" = " * 80)

    # Basic examples
    example_basic_training()
    example_advanced_training()
    example_cross_validation()
    example_automated_model_selection()
    example_live_dashboard()

    # Advanced examples (async)
    print("\n🔄 Running async examples...")
    try:
        asyncio.run(example_ensemble_training())
    except Exception as e:
        print(f"Ensemble training example failed: {e}")

    # Hyperparameter optimization (may take longer)
    try:
        example_hyperparameter_optimization()
    except Exception as e:
        print(f"Hyperparameter optimization example failed: {e}")

    print("\n" + " = " * 80)
    print("✅ All examples completed!")
    print(" = " * 80)

if __name__ == "__main__":
    run_all_examples()