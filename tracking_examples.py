# Enterprise Tracking System Examples
# tracking_examples.py
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from tracking import start_experiment, tracker
from tracking_integration import (
                from tracking_integration import production_tracker
        import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
    import sys
import time
import warnings
"""
Comprehensive examples showing how to use the enterprise tracking system
for real - world ML projects
"""

warnings.filterwarnings('ignore')

    start_production_monitoring, 
    start_data_pipeline, 
    deploy_model, 
    log_prediction
)


console = Console()

def example_1_basic_experiment():
    """
    Example 1: Basic ML experiment tracking
    """
    console.print(Panel("🧪 Example 1: Basic ML Experiment", border_style = "blue"))

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Start experiment
    with start_experiment("basic_classification", "random_forest_baseline") as exp:

        # Log dataset information
        exp.log_params({
            "dataset_size": len(X), 
            "n_features": X.shape[1], 
            "n_classes": len(np.unique(y)), 
            "train_size": len(X_train), 
            "test_size": len(X_test), 
            "random_state": 42
        })

        # Train model
        model = RandomForestClassifier(n_estimators = 100, random_state = 42)
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Log model parameters
        exp.log_params({
            "n_estimators": model.n_estimators, 
            "max_depth": model.max_depth, 
            "min_samples_split": model.min_samples_split, 
            "min_samples_leaf": model.min_samples_leaf
        })

        # Log performance metrics
        exp.log_metrics({
            "accuracy": accuracy, 
            "training_time_seconds": training_time, 
            "model_size_features": X.shape[1], 
            "prediction_confidence_mean": np.mean(np.max(y_proba, axis = 1))
        })

        # Log model
        exp.log_model(model, "random_forest_baseline")

        # Create and log confusion matrix plot
        plt.figure(figsize = (8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        exp.log_figure(plt.gcf(), "confusion_matrix")
        plt.close()

        # Create and log feature importance plot
        plt.figure(figsize = (10, 6))
        feature_importance = model.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
        plt.bar(feature_names, feature_importance)
        plt.title('Feature Importance')
        plt.xticks(rotation = 45)
        exp.log_figure(plt.gcf(), "feature_importance")
        plt.close()

        # Log classification report as artifact
        report = classification_report(y_test, y_pred, output_dict = True)
        with open('classification_report.json', 'w') as f:
            json.dump(report, f, indent = 2)
        exp.log_artifact('classification_report.json')

        console.print(f"✅ Basic experiment completed! Accuracy: {accuracy:.3f}")

def example_2_hyperparameter_tuning():
    """
    Example 2: Hyperparameter tuning with experiment tracking
    """
    console.print(Panel("🔧 Example 2: Hyperparameter Tuning", border_style = "green"))

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(500, 8)
    y = ((X[:, 0] + X[:, 1] + np.random.randn(500) * 0.1) > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200], 
        'max_depth': [3, 5, 10, None], 
        'min_samples_split': [2, 5, 10]
    }

    with start_experiment("hyperparameter_tuning", "grid_search_rf") as exp:

        # Log experiment setup
        exp.log_params({
            "algorithm": "GridSearchCV", 
            "base_model": "RandomForestClassifier", 
            "cv_folds": 5, 
            "param_grid_size": len(param_grid['n_estimators']) *
                             len(param_grid['max_depth']) *
                             len(param_grid['min_samples_split']), 
            "dataset_size": len(X), 
            "scoring": "accuracy"
        })

        # Perform grid search with progress tracking
        rf = RandomForestClassifier(random_state = 42)
        grid_search = GridSearchCV(rf, param_grid, cv = 5, scoring = 'accuracy', n_jobs = -1)

        console.print("🔍 Running grid search...")
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time

        # Log search results
        exp.log_metrics({
            "best_cv_score": grid_search.best_score_, 
            "search_time_seconds": search_time, 
            "n_combinations_tested": len(grid_search.cv_results_['params'])
        })

        # Log best parameters
        exp.log_params(grid_search.best_params_, prefix = "best_")

        # Test best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        exp.log_metrics({
            "test_accuracy": test_accuracy, 
            "cv_test_gap": abs(grid_search.best_score_ - test_accuracy)
        })

        # Log detailed results
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv('grid_search_results.csv', index = False)
        exp.log_artifact('grid_search_results.csv')

        # Create parameter vs performance plot
        plt.figure(figsize = (12, 8))

        # Plot n_estimators effect
        plt.subplot(2, 2, 1)
        estimator_scores = results_df.groupby('param_n_estimators')['mean_test_score'].mean()
        plt.plot(estimator_scores.index, estimator_scores.values, 'o - ')
        plt.title('N_estimators vs CV Score')
        plt.xlabel('N_estimators')
        plt.ylabel('CV Score')

        # Plot max_depth effect
        plt.subplot(2, 2, 2)
        depth_scores = results_df.groupby('param_max_depth')['mean_test_score'].mean()
        plt.plot(range(len(depth_scores)), depth_scores.values, 'o - ')
        plt.title('Max_depth vs CV Score')
        plt.xlabel('Max_depth')
        plt.ylabel('CV Score')
        plt.xticks(range(len(depth_scores)), depth_scores.index)

        # Plot min_samples_split effect
        plt.subplot(2, 2, 3)
        split_scores = results_df.groupby('param_min_samples_split')['mean_test_score'].mean()
        plt.plot(split_scores.index, split_scores.values, 'o - ')
        plt.title('Min_samples_split vs CV Score')
        plt.xlabel('Min_samples_split')
        plt.ylabel('CV Score')

        # Plot score distribution
        plt.subplot(2, 2, 4)
        plt.hist(results_df['mean_test_score'], bins = 20, alpha = 0.7)
        plt.axvline(grid_search.best_score_, color = 'red', linestyle = ' -  - ', label = 'Best Score')
        plt.title('CV Score Distribution')
        plt.xlabel('CV Score')
        plt.ylabel('Frequency')
        plt.legend()

        plt.tight_layout()
        exp.log_figure(plt.gcf(), "hyperparameter_analysis")
        plt.close()

        # Log best model
        exp.log_model(best_model, "best_random_forest")

        console.print(f"✅ Hyperparameter tuning completed! Best CV Score: {grid_search.best_score_:.3f}")

def example_3_data_pipeline_tracking():
    """
    Example 3: Data pipeline tracking
    """
    console.print(Panel("📊 Example 3: Data Pipeline Tracking", border_style = "yellow"))

    # Simulate data pipeline
    with start_data_pipeline("market_data_pipeline", "simulated_source", 10000) as pipeline:

        # Stage 1: Data Extraction
        console.print("📥 Stage 1: Data Extraction")
        time.sleep(1)  # Simulate extraction time
        raw_data_size = 9800  # Simulate some data loss
        pipeline.log_stage("extraction", raw_data_size, errors = 200, duration_seconds = 15.3)

        # Stage 2: Data Validation
        console.print("🔍 Stage 2: Data Validation")
        time.sleep(0.5)
        validation_errors = 150
        valid_data_size = raw_data_size - validation_errors
        pipeline.log_stage("validation", valid_data_size, errors = validation_errors, duration_seconds = 8.7)

        # Stage 3: Data Cleaning
        console.print("🧹 Stage 3: Data Cleaning")
        time.sleep(0.8)
        cleaned_data_size = valid_data_size - 50  # Remove outliers
        pipeline.log_stage("cleaning", cleaned_data_size, errors = 50, duration_seconds = 12.1)

        # Stage 4: Feature Engineering
        console.print("⚙️ Stage 4: Feature Engineering")
        time.sleep(1.2)
        feature_errors = 20
        final_data_size = cleaned_data_size - feature_errors
        pipeline.log_stage("feature_engineering", final_data_size, errors = feature_errors, duration_seconds = 18.9)

        # Stage 5: Data Quality Checks
        console.print("✅ Stage 5: Quality Checks")
        time.sleep(0.3)
        pipeline.log_stage("quality_checks", final_data_size, errors = 0, duration_seconds = 5.2)

        # Log data quality metrics
        pipeline.log_data_quality({
            "completeness": final_data_size / 10000,  # Original expected size
            "accuracy": 0.98, 
            "consistency": 0.96, 
            "timeliness": 0.95, 
            "validity": 0.97
        })

        # Complete pipeline
        pipeline.complete_pipeline(final_data_size, success = True)

        console.print(f"✅ Data pipeline completed! Processed {final_data_size:, } records")

def example_4_production_monitoring():
    """
    Example 4: Production model monitoring
    """
    console.print(Panel("🚀 Example 4: Production Monitoring", border_style = "red"))

    # Deploy model for monitoring
    deployment_config = {
        "environment": "production", 
        "version": "1.0.0", 
        "deployment_type": "real - time", 
        "scaling_config": {
            "min_replicas": 2, 
            "max_replicas": 10, 
            "cpu_threshold": 70
        }, 
        "monitoring_enabled": True
    }

    deployment_id = deploy_model("trading_classifier", "1.0.0", deployment_config)
    console.print(f"📦 Model deployed with ID: {deployment_id}")

    # Start production monitoring
    start_production_monitoring("trading_classifier", deployment_id)

    # Simulate production predictions
    console.print("🔄 Simulating production predictions...")

    with Progress(
        TextColumn("[progress.description]{task.description}"), 
        BarColumn(), 
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
        console = console
    ) as progress:

        prediction_task = progress.add_task("Making predictions...", total = 100)

        for i in range(100):
            # Simulate prediction input
            input_data = {
                "feature_1": np.random.randn(), 
                "feature_2": np.random.randn(), 
                "feature_3": np.random.randn(), 
                "timestamp": datetime.now().isoformat()
            }

            # Simulate prediction
            start_time = time.time()
            prediction = np.random.choice([0, 1])  # Random prediction
            confidence = np.random.uniform(0.6, 0.95)  # Random confidence
            latency_ms = (time.time() - start_time) * 1000 + np.random.uniform(10, 100)

            # Log prediction
            log_prediction(
                deployment_id = deployment_id, 
                input_data = input_data, 
                prediction = prediction, 
                confidence = confidence, 
                latency_ms = latency_ms
            )

            # Simulate occasional trading results
            if i % 10 == 0:
                trade_result = {
                    "pnl": np.random.uniform( - 500, 1000), 
                    "return_pct": np.random.uniform( - 0.02, 0.05), 
                    "position_size": np.random.uniform(1000, 10000), 
                    "execution_time": datetime.now().isoformat()
                }
                production_tracker.log_trade_result(deployment_id, trade_result)

            progress.update(prediction_task, advance = 1)
            time.sleep(0.01)  # Small delay to simulate real - time

    console.print("✅ Production monitoring simulation completed!")

    # Show production summary
    summary = production_tracker.get_production_summary(deployment_id)
    if summary:
        console.print(f"📊 Production Summary:")
        for key, value in summary.items():
            console.print(f"   {key}: {value}")

def example_5_model_comparison():
    """
    Example 5: Comparing multiple models
    """
    console.print(Panel("⚖️ Example 5: Model Comparison", border_style = "purple"))

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(800, 6)
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Models to compare
    models = {
        "RandomForest": RandomForestClassifier(n_estimators = 100, random_state = 42), 
        "RandomForest_Tuned": RandomForestClassifier(n_estimators = 200, max_depth = 10, random_state = 42), 
        "RandomForest_Simple": RandomForestClassifier(n_estimators = 50, max_depth = 5, random_state = 42)
    }

    model_results = []

    for model_name, model in models.items():
        console.print(f"🔄 Training {model_name}...")

        with start_experiment("model_comparison", f"{model_name}_comparison") as exp:

            # Log model configuration
            exp.log_params({
                "model_type": model.__class__.__name__, 
                "model_name": model_name, 
                **{f"param_{k}": v for k, v in model.get_params().items()}
            })

            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Log metrics
            metrics = {
                "accuracy": accuracy, 
                "training_time": training_time, 
                "avg_confidence": np.mean(np.max(y_proba, axis = 1)), 
                "model_complexity": getattr(model, 'n_estimators', 1) * getattr(model, 'max_depth', 1)
            }

            exp.log_metrics(metrics)
            exp.log_model(model, f"model_{model_name}")

            # Store results for comparison
            model_results.append({
                "model_name": model_name, 
                "run_id": exp.run_id, 
                **metrics
            })

    # Create comparison visualization
    plt.figure(figsize = (15, 10))

    # Accuracy comparison
    plt.subplot(2, 3, 1)
    accuracies = [r["accuracy"] for r in model_results]
    model_names = [r["model_name"] for r in model_results]
    plt.bar(model_names, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation = 45)

    # Training time comparison
    plt.subplot(2, 3, 2)
    times = [r["training_time"] for r in model_results]
    plt.bar(model_names, times)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation = 45)

    # Confidence comparison
    plt.subplot(2, 3, 3)
    confidences = [r["avg_confidence"] for r in model_results]
    plt.bar(model_names, confidences)
    plt.title('Average Confidence Comparison')
    plt.ylabel('Confidence')
    plt.xticks(rotation = 45)

    # Accuracy vs Training Time scatter
    plt.subplot(2, 3, 4)
    plt.scatter(times, accuracies)
    for i, name in enumerate(model_names):
        plt.annotate(name, (times[i], accuracies[i]), xytext = (5, 5), textcoords = 'offset points')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Training Time')

    # Model complexity comparison
    plt.subplot(2, 3, 5)
    complexities = [r["model_complexity"] for r in model_results]
    plt.bar(model_names, complexities)
    plt.title('Model Complexity Comparison')
    plt.ylabel('Complexity Score')
    plt.xticks(rotation = 45)

    # Performance radar chart
    plt.subplot(2, 3, 6)
    # Normalize metrics for radar chart
    norm_acc = [(a - min(accuracies)) / (max(accuracies) - min(accuracies)) for a in accuracies]
    norm_time = [1 - (t - min(times)) / (max(times) - min(times)) for t in times]  # Inverted (lower is better)
    norm_conf = [(c - min(confidences)) / (max(confidences) - min(confidences)) for c in confidences]

    for i, name in enumerate(model_names):
        values = [norm_acc[i], norm_time[i], norm_conf[i]]
        angles = np.linspace(0, 2*np.pi, len(values), endpoint = False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        plt.plot(angles, values, 'o - ', label = name)
        plt.fill(angles, values, alpha = 0.25)

    plt.xticks(angles[: - 1], ['Accuracy', 'Speed', 'Confidence'])
    plt.title('Model Performance Radar')
    plt.legend()

    plt.tight_layout()

    # Log comparison to a meta - experiment
    with start_experiment("model_comparison", "comparison_summary") as exp:
        exp.log_params({
            "models_compared": len(models), 
            "dataset_size": len(X), 
            "comparison_date": datetime.now().isoformat()
        })

        # Log best model info
        best_model = max(model_results, key = lambda x: x["accuracy"])
        exp.log_params(best_model, prefix = "best_")

        exp.log_figure(plt.gcf(), "model_comparison_analysis")

        # Save results table
        results_df = pd.DataFrame(model_results)
        results_df.to_csv('model_comparison_results.csv', index = False)
        exp.log_artifact('model_comparison_results.csv')

    plt.close()

    console.print("✅ Model comparison completed!")
    console.print(f"🏆 Best model: {best_model['model_name']} (Accuracy: {best_model['accuracy']:.3f})")

def run_all_examples():
    """
    Run all examples in sequence
    """
    console.print(Panel(
        "🚀 Running Enterprise Tracking System Examples\n"
        "This will demonstrate all tracking capabilities", 
        title = "Enterprise ML Tracking Demo", 
        border_style = "bold blue"
    ))

    examples = [
        ("Basic Experiment", example_1_basic_experiment), 
        ("Hyperparameter Tuning", example_2_hyperparameter_tuning), 
        ("Data Pipeline Tracking", example_3_data_pipeline_tracking), 
        ("Production Monitoring", example_4_production_monitoring), 
        ("Model Comparison", example_5_model_comparison)
    ]

    for name, example_func in examples:
        try:
            console.print(f"\n🔄 Running: {name}")
            example_func()
            console.print(f"✅ Completed: {name}")
        except Exception as e:
            console.print(f"❌ Failed: {name} - {str(e)}")

        console.print(" - " * 80)

    console.print(Panel(
        "🎉 All examples completed!\n\n"
        "Check your tracking directory for:\n"
        "• Experiment runs and metadata\n"
        "• Model artifacts and plots\n"
        "• Data pipeline logs\n"
        "• Production monitoring data\n\n"
        "Use the CLI to explore results:\n"
        "  python tracking_cli.py list - experiments\n"
        "  python tracking_cli.py generate - report", 
        title = "Demo Complete", 
        border_style = "bold green"
    ))

if __name__ == "__main__":
    # You can run individual examples or all at once

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples = {
            "1": example_1_basic_experiment, 
            "2": example_2_hyperparameter_tuning, 
            "3": example_3_data_pipeline_tracking, 
            "4": example_4_production_monitoring, 
            "5": example_5_model_comparison, 
            "all": run_all_examples
        }

        if example_num in examples:
            examples[example_num]()
        else:
            console.print("❌ Invalid example number. Use 1 - 5 or 'all'")
    else:
        console.print("Usage: python tracking_examples.py [1 - 5|all]")
        console.print("Examples:")
        console.print("  python tracking_examples.py 1    # Basic experiment")
        console.print("  python tracking_examples.py all  # Run all examples")