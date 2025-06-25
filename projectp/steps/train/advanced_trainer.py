from .main_trainer import MainTrainer
from .utils import TrainingUtils, ConfigValidator
from pathlib import Path
from projectp.pro_log import pro_log
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from typing import Dict, Any, Optional, List, Callable
import asyncio
import concurrent.futures
import numpy as np
            import optuna
"""
Advanced Training Pipeline
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
Enhanced version with advanced features for production use
"""


console = Console()

class AdvancedTrainer(MainTrainer):
    """Advanced training pipeline with enhanced features"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.validator = ConfigValidator()
        self.parallel_executor = concurrent.futures.ThreadPoolExecutor(max_workers = 4)

    async def run_ensemble_training(self, 
                                   data_path: Optional[str] = None, 
                                   target_auc: float = 70.0, 
                                   ensemble_size: int = 3) -> Dict[str, Any]:
        """
        Run ensemble training with multiple models in parallel
        """
        console.print("\n[bold blue]🎯 Starting Ensemble Training Pipeline[/bold blue]")

        # Validate configuration
        if not self.validator.validate_ensemble_config(ensemble_size, target_auc):
            return {'success': False, 'error': 'Invalid ensemble configuration'}

        # Prepare different model configurations
        model_configs = self._generate_ensemble_configs(ensemble_size)

        # Run parallel training
        tasks = []
        for i, config in enumerate(model_configs):
            trainer = MainTrainer(config)
            task = asyncio.create_task(
                self._async_train_single_model(trainer, data_path, target_auc, i + 1)
            )
            tasks.append(task)

        # Wait for all models to complete
        results = await asyncio.gather(*tasks, return_exceptions = True)

        # Process ensemble results
        return self._process_ensemble_results(results, target_auc)

    def run_hyperparameter_optimization(self, 
                                       data_path: Optional[str] = None, 
                                       target_auc: float = 70.0, 
                                       n_trials: int = 50) -> Dict[str, Any]:
        """
        Run advanced hyperparameter optimization
        """
        console.print("\n[bold blue]🔧 Starting Hyperparameter Optimization[/bold blue]")

        try:

            def objective(trial):
                # Define hyperparameter search space
                config = {
                    'model_type': trial.suggest_categorical('model_type', ['catboost', 'xgboost', 'lightgbm']), 
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), 
                    'max_depth': trial.suggest_int('max_depth', 3, 10), 
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 
                    'random_state': 42
                }

                trainer = MainTrainer(config)
                results = trainer.run_full_pipeline(data_path = data_path, target_auc = target_auc, max_iterations = 1)
                return results.get('best_auc', 0.0)

            # Create and run study
            study = optuna.create_study(direction = 'maximize')
            study.optimize(objective, n_trials = n_trials, show_progress_bar = True)

            # Train final model with best parameters
            best_config = study.best_params
            final_trainer = MainTrainer(best_config)
            final_results = final_trainer.run_full_pipeline(data_path = data_path, target_auc = target_auc)

            return {
                'success': True, 
                'best_params': best_config, 
                'best_auc': study.best_value, 
                'final_results': final_results, 
                'optimization_trials': n_trials
            }

        except ImportError:
            console.print("[yellow]Optuna not available, falling back to grid search[/yellow]")
            return self._run_grid_search(data_path, target_auc)
        except Exception as e:
            pro_log.error(f"Hyperparameter optimization failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def run_cross_validation_training(self, 
                                     data_path: Optional[str] = None, 
                                     target_auc: float = 70.0, 
                                     cv_folds: int = 5) -> Dict[str, Any]:
        """
        Run cross - validation training for robust model evaluation
        """
        console.print(f"\n[bold blue]📊 Starting {cv_folds} - Fold Cross - Validation Training[/bold blue]")

        cv_results = []
        best_fold_auc = 0.0
        best_fold_model = None

        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"), 
            BarColumn(), 
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
            TimeElapsedColumn(), 
            console = console
        ) as progress:

            cv_task = progress.add_task("Cross - Validation Progress", total = cv_folds)

            for fold in range(cv_folds):
                progress.update(cv_task, description = f"Training fold {fold + 1}/{cv_folds}...")

                # Create fold - specific configuration
                fold_config = self.config.copy()
                fold_config['cv_fold'] = fold
                fold_config['random_state'] = 42 + fold

                # Train on this fold
                fold_trainer = MainTrainer(fold_config)
                fold_results = fold_trainer.run_full_pipeline(
                    data_path = data_path, 
                    target_auc = target_auc, 
                    max_iterations = 3  # Fewer iterations for CV
                )

                fold_auc = fold_results.get('best_auc', 0.0)
                cv_results.append({
                    'fold': fold + 1, 
                    'auc': fold_auc, 
                    'results': fold_results
                })

                # Track best fold
                if fold_auc > best_fold_auc:
                    best_fold_auc = fold_auc
                    best_fold_model = fold_results

                progress.update(cv_task, completed = fold + 1)

                console.print(f"[cyan]Fold {fold + 1}: AUC = {fold_auc:.2f}%[/cyan]")

        # Calculate CV statistics
        cv_aucs = [result['auc'] for result in cv_results]
        cv_mean = np.mean(cv_aucs)
        cv_std = np.std(cv_aucs)

        # Display CV summary
        self._display_cv_summary(cv_results, cv_mean, cv_std, target_auc)

        return {
            'success': True, 
            'cv_results': cv_results, 
            'cv_mean_auc': cv_mean, 
            'cv_std_auc': cv_std, 
            'best_fold_auc': best_fold_auc, 
            'best_fold_model': best_fold_model, 
            'target_achieved': cv_mean >= target_auc
        }

    def run_automated_model_selection(self, 
                                     data_path: Optional[str] = None, 
                                     target_auc: float = 70.0) -> Dict[str, Any]:
        """
        Automatically select best model from multiple algorithms
        """
        console.print("\n[bold blue]🤖 Starting Automated Model Selection[/bold blue]")

        # Define model candidates
        model_candidates = [
            {'name': 'CatBoost', 'config': {'model_type': 'catboost'}}, 
            {'name': 'XGBoost', 'config': {'model_type': 'xgboost'}}, 
            {'name': 'LightGBM', 'config': {'model_type': 'lightgbm'}}, 
            {'name': 'Random Forest', 'config': {'model_type': 'random_forest'}}, 
        ]

        model_results = []
        best_model = None
        best_auc = 0.0

        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"), 
            BarColumn(), 
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
            TimeElapsedColumn(), 
            console = console
        ) as progress:

            selection_task = progress.add_task("Model Selection", total = len(model_candidates))

            for i, candidate in enumerate(model_candidates):
                progress.update(selection_task, description = f"Testing {candidate['name']}...")

                # Configure model
                model_config = self.config.copy()
                model_config.update(candidate['config'])

                # Train model
                trainer = MainTrainer(model_config)
                results = trainer.run_full_pipeline(
                    data_path = data_path, 
                    target_auc = target_auc, 
                    max_iterations = 3
                )

                model_auc = results.get('best_auc', 0.0)
                model_results.append({
                    'name': candidate['name'], 
                    'auc': model_auc, 
                    'config': model_config, 
                    'results': results
                })

                # Track best model
                if model_auc > best_auc:
                    best_auc = model_auc
                    best_model = {
                        'name': candidate['name'], 
                        'config': model_config, 
                        'results': results
                    }

                progress.update(selection_task, completed = i + 1)
                console.print(f"[cyan]{candidate['name']}: AUC = {model_auc:.2f}%[/cyan]")

        # Display selection summary
        self._display_model_selection_summary(model_results, best_model, target_auc)

        return {
            'success': True, 
            'model_results': model_results, 
            'best_model': best_model, 
            'best_auc': best_auc, 
            'target_achieved': best_auc >= target_auc
        }

    def _generate_ensemble_configs(self, ensemble_size: int) -> List[Dict[str, Any]]:
        """Generate diverse configurations for ensemble"""
        base_config = self.config.copy()
        configs = []

        model_types = ['catboost', 'xgboost', 'lightgbm']

        for i in range(ensemble_size):
            config = base_config.copy()
            config['model_type'] = model_types[i % len(model_types)]
            config['random_state'] = 42 + i * 10
            config['ensemble_id'] = i + 1
            configs.append(config)

        return configs

    async def _async_train_single_model(self, trainer: MainTrainer, data_path: str, target_auc: float, model_id: int):
        """Train a single model asynchronously"""
        loop = asyncio.get_event_loop()

        def train():
            return trainer.run_full_pipeline(
                data_path = data_path, 
                target_auc = target_auc, 
                max_iterations = 3
            )

        result = await loop.run_in_executor(self.parallel_executor, train)
        result['model_id'] = model_id
        return result

    def _process_ensemble_results(self, results: List[Dict[str, Any]], target_auc: float) -> Dict[str, Any]:
        """Process ensemble training results"""
        successful_models = [r for r in results if isinstance(r, dict) and r.get('success', False)]

        if not successful_models:
            return {'success': False, 'error': 'No models trained successfully'}

        # Calculate ensemble metrics
        aucs = [r.get('best_auc', 0.0) for r in successful_models]
        ensemble_auc = np.mean(aucs)
        ensemble_std = np.std(aucs)

        return {
            'success': True, 
            'ensemble_auc': ensemble_auc, 
            'ensemble_std': ensemble_std, 
            'individual_results': successful_models, 
            'models_trained': len(successful_models), 
            'target_achieved': ensemble_auc >= target_auc
        }

    def _run_grid_search(self, data_path: str, target_auc: float) -> Dict[str, Any]:
        """Fallback grid search when Optuna is not available"""
        console.print("[yellow]Running basic grid search...[/yellow]")

        param_grid = {
            'model_type': ['catboost', 'xgboost'], 
            'learning_rate': [0.1, 0.2], 
            'max_depth': [6, 8]
        }

        best_auc = 0.0
        best_config = None

        # Simple grid search implementation
        for model_type in param_grid['model_type']:
            for lr in param_grid['learning_rate']:
                for depth in param_grid['max_depth']:
                    config = {
                        'model_type': model_type, 
                        'learning_rate': lr, 
                        'max_depth': depth, 
                        'random_state': 42
                    }

                    trainer = MainTrainer(config)
                    results = trainer.run_full_pipeline(data_path = data_path, target_auc = target_auc, max_iterations = 1)
                    auc = results.get('best_auc', 0.0)

                    if auc > best_auc:
                        best_auc = auc
                        best_config = config

        return {
            'success': True, 
            'best_params': best_config, 
            'best_auc': best_auc, 
            'method': 'grid_search'
        }

    def _display_cv_summary(self, cv_results: List[Dict], mean_auc: float, std_auc: float, target_auc: float):
        """Display cross - validation summary"""
        table = Table(title = "📊 Cross - Validation Results", show_header = True, header_style = "bold blue")
        table.add_column("Fold", style = "cyan", width = 8)
        table.add_column("AUC (%)", style = "magenta", width = 12)
        table.add_column("Status", style = "green", width = 12)

        for result in cv_results:
            auc = result['auc']
            status = "✅ Target" if auc >= target_auc else "📊 Training"
            table.add_row(str(result['fold']), f"{auc:.2f}", status)

        table.add_row("", "", "")
        table.add_row("Mean", f"{mean_auc:.2f}", "✅ Success" if mean_auc >= target_auc else "⚠️ Below Target")
        table.add_row("Std Dev", f"{std_auc:.2f}", "")

        console.print("\n")
        console.print(table)
        console.print("\n")

    def _display_model_selection_summary(self, model_results: List[Dict], best_model: Dict, target_auc: float):
        """Display model selection summary"""
        table = Table(title = "🤖 Model Selection Results", show_header = True, header_style = "bold blue")
        table.add_column("Model", style = "cyan", width = 15)
        table.add_column("AUC (%)", style = "magenta", width = 12)
        table.add_column("Status", style = "green", width = 15)

        for result in sorted(model_results, key = lambda x: x['auc'], reverse = True):
            auc = result['auc']
            status = "🏆 Best" if result['name'] == best_model['name'] else "📊 Candidate"
            if auc >= target_auc:
                status += " ✅"
            table.add_row(result['name'], f"{auc:.2f}", status)

        console.print("\n")
        console.print(table)
        console.print("\n")

        if best_model:
            console.print(Panel(
                f"[bold green]🏆 Best Model: {best_model['name']} (AUC: {best_model['results'].get('best_auc', 0):.2f}%)[/bold green]", 
                style = "green"
            ))

# Convenience functions for advanced training
async def run_ensemble_train(config: Optional[Dict[str, Any]] = None, **kwargs):
    """Run ensemble training"""
    trainer = AdvancedTrainer(config)
    return await trainer.run_ensemble_training(**kwargs)

def run_hyperopt_train(config: Optional[Dict[str, Any]] = None, **kwargs):
    """Run hyperparameter optimization training"""
    trainer = AdvancedTrainer(config)
    return trainer.run_hyperparameter_optimization(**kwargs)

def run_cv_train(config: Optional[Dict[str, Any]] = None, **kwargs):
    """Run cross - validation training"""
    trainer = AdvancedTrainer(config)
    return trainer.run_cross_validation_training(**kwargs)

def run_automl_train(config: Optional[Dict[str, Any]] = None, **kwargs):
    """Run automated model selection"""
    trainer = AdvancedTrainer(config)
    return trainer.run_automated_model_selection(**kwargs)