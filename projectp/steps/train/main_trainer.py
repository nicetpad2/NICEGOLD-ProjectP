"""
Main Training Pipeline
=====================
Orchestrates the complete training workflow
"""

import sys
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from projectp.pro_log import pro_log
from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .saver import ModelSaver
from .utils import TrainingUtils

console = Console()

class MainTrainer:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.results = {}
        self.utils = TrainingUtils()
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator(config)
        self.saver = ModelSaver(config)
        
    def run_full_pipeline(self, 
                         data_path: Optional[str] = None,
                         target_auc: float = 70.0,
                         max_iterations: int = 5) -> Dict[str, Any]:
        """
        Run complete training pipeline with AUC target
        
        Args:
            data_path: Path to training data
            target_auc: Target AUC score to achieve
            max_iterations: Maximum training iterations
            
        Returns:
            Dict containing training results and metrics
        """
        console.print("\n[bold blue]🚀 Starting ML Training Pipeline[/bold blue]")
        
        pipeline_start_time = time.time()
        best_auc = 0.0
        best_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Task tracking
            main_task = progress.add_task("Pipeline Progress", total=100)
            
            try:
                # Step 1: Data Processing (20%)
                progress.update(main_task, description="Processing data...", completed=10)
                data_results = self._process_data(data_path)
                if not data_results['success']:
                    return self._create_error_result("Data processing failed", data_results)
                progress.update(main_task, completed=20)
                
                # Step 2: Feature Engineering (40%)
                progress.update(main_task, description="Engineering features...", completed=25)
                feature_results = self._engineer_features(data_results['data'])
                if not feature_results['success']:
                    return self._create_error_result("Feature engineering failed", feature_results)
                progress.update(main_task, completed=40)
                
                # Step 3: Model Training Loop (60-80%)
                progress.update(main_task, description="Training models...", completed=45)
                
                for iteration in range(max_iterations):
                    iteration_start = time.time()
                    
                    # Train model
                    training_results = self._train_model(feature_results['data'])
                    if not training_results['success']:
                        console.print(f"[yellow]Warning: Training iteration {iteration + 1} failed[/yellow]")
                        continue
                    
                    # Evaluate model
                    eval_results = self._evaluate_model(training_results['model'], feature_results['data'])
                    
                    current_auc = eval_results.get('auc', 0.0)
                    
                    # Update progress
                    progress_value = 45 + (iteration + 1) * (35 / max_iterations)
                    progress.update(main_task, completed=progress_value)
                    
                    # Check if we achieved target AUC
                    if current_auc > best_auc:
                        best_auc = current_auc
                        best_results = {
                            'model': training_results['model'],
                            'evaluation': eval_results,
                            'iteration': iteration + 1,
                            'training_time': time.time() - iteration_start
                        }
                    
                    # Display results
                    self._display_iteration_results(iteration + 1, current_auc, target_auc)
                    
                    # Check if target reached
                    if current_auc >= target_auc:
                        console.print(f"[bold green]🎯 Target AUC {target_auc}% achieved! (AUC: {current_auc:.2f}%)[/bold green]")
                        break
                
                progress.update(main_task, completed=80)
                
                # Step 4: Model Saving (100%)
                progress.update(main_task, description="Saving best model...", completed=85)
                if best_results:
                    save_results = self._save_model(best_results['model'], best_results['evaluation'])
                    best_results['save_results'] = save_results
                progress.update(main_task, completed=100)
                
                # Final results
                pipeline_time = time.time() - pipeline_start_time
                final_results = self._create_final_results(best_results, pipeline_time, target_auc)
                
                self._display_final_summary(final_results)
                return final_results
                
            except Exception as e:
                error_msg = f"Pipeline failed with error: {str(e)}"
                pro_log.error(error_msg)
                console.print(f"[bold red]❌ {error_msg}[/bold red]")
                return self._create_error_result(error_msg, {'exception': str(e)})
    
    def _process_data(self, data_path: Optional[str]) -> Dict[str, Any]:
        """Process training data"""
        try:
            data = self.data_processor.load_and_prepare_data(data_path)
            return {'success': True, 'data': data}
        except Exception as e:
            pro_log.error(f"Data processing error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _engineer_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features for training"""
        try:
            engineered_data = self.feature_engineer.engineer_features(data)
            return {'success': True, 'data': engineered_data}
        except Exception as e:
            pro_log.error(f"Feature engineering error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _train_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train a model"""
        try:
            model = self.model_trainer.train(data)
            return {'success': True, 'model': model}
        except Exception as e:
            pro_log.error(f"Model training error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_model(self, model: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained model"""
        try:
            return self.evaluator.evaluate(model, data)
        except Exception as e:
            pro_log.error(f"Model evaluation error: {str(e)}")
            return {'auc': 0.0, 'error': str(e)}
    
    def _save_model(self, model: Any, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Save the trained model"""
        try:
            return self.saver.save_model(model, evaluation)
        except Exception as e:
            pro_log.error(f"Model saving error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _display_iteration_results(self, iteration: int, auc: float, target_auc: float):
        """Display results for current iteration"""
        status = "🎯" if auc >= target_auc else "📊"
        color = "green" if auc >= target_auc else "yellow" if auc >= target_auc * 0.8 else "red"
        
        console.print(f"{status} [bold {color}]Iteration {iteration}: AUC = {auc:.2f}%[/bold {color}] (Target: {target_auc}%)")
    
    def _display_final_summary(self, results: Dict[str, Any]):
        """Display final training summary"""
        table = Table(title="🏁 Training Pipeline Summary", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="magenta", width=30)
        
        table.add_row("Best AUC", f"{results.get('best_auc', 0):.2f}%")
        table.add_row("Target AUC", f"{results.get('target_auc', 0):.2f}%")
        table.add_row("Target Achieved", "✅ Yes" if results.get('target_achieved', False) else "❌ No")
        table.add_row("Best Iteration", str(results.get('best_iteration', 'N/A')))
        table.add_row("Total Time", f"{results.get('total_time', 0):.2f}s")
        table.add_row("Model Saved", "✅ Yes" if results.get('model_saved', False) else "❌ No")
        
        console.print("\n")
        console.print(table)
        console.print("\n")
        
        # Success/failure message
        if results.get('target_achieved', False):
            console.print(Panel(
                f"[bold green]🎉 SUCCESS! Target AUC of {results.get('target_auc', 0):.1f}% achieved with {results.get('best_auc', 0):.2f}%[/bold green]",
                style="green"
            ))
        else:
            console.print(Panel(
                f"[bold yellow]⚠️  Target not reached. Best AUC: {results.get('best_auc', 0):.2f}% (Target: {results.get('target_auc', 0):.1f}%)[/bold yellow]",
                style="yellow"
            ))
    
    def _create_final_results(self, best_results: Dict[str, Any], pipeline_time: float, target_auc: float) -> Dict[str, Any]:
        """Create final results dictionary"""
        best_auc = best_results.get('evaluation', {}).get('auc', 0.0) if best_results else 0.0
        
        return {
            'success': True,
            'best_auc': best_auc,
            'target_auc': target_auc,
            'target_achieved': best_auc >= target_auc,
            'best_iteration': best_results.get('iteration', 0) if best_results else 0,
            'total_time': pipeline_time,
            'model_saved': best_results.get('save_results', {}).get('success', False) if best_results else False,
            'results': best_results
        }
    
    def _create_error_result(self, message: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            'success': False,
            'error': message,
            'details': details,
            'best_auc': 0.0,
            'target_achieved': False
        }

def run_train(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Main training function for backward compatibility
    
    Args:
        config: Training configuration
        **kwargs: Additional training parameters
        
    Returns:
        Training results
    """
    trainer = MainTrainer(config)
    return trainer.run_full_pipeline(**kwargs)

# For direct execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Training Pipeline")
    parser.add_argument('--data_path', type=str, help='Path to training data')
    parser.add_argument('--target_auc', type=float, default=70.0, help='Target AUC score')
    parser.add_argument('--max_iterations', type=int, default=5, help='Maximum training iterations')
    
    args = parser.parse_args()
    
    trainer = MainTrainer()
    results = trainer.run_full_pipeline(
        data_path=args.data_path,
        target_auc=args.target_auc,
        max_iterations=args.max_iterations
    )
    
    # Print final AUC for task integration
    if results.get('success', False):
        print(f"AUC: {results.get('best_auc', 0.0)}")
    else:
        print("AUC: 0.0")
