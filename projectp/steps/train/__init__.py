    from .advanced_trainer import (
    from .data_processor import DataProcessor
    from .evaluator import ModelEvaluator
    from .feature_engineer import FeatureEngineer
    from .main_trainer import run_train, MainTrainer
    from .model_trainer import ModelTrainer
    from .monitor import (
    from .saver import ModelSaver
    from .utils import TrainingUtils, ConfigValidator
    from projectp.steps.train import DataProcessor, ModelTrainer
    from projectp.steps.train import run_train
from typing import Dict, Any, Optional
import warnings
"""
Training Module - Modular Structure
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
Sub - modules for better maintainability and future development

Modules:
- data_processor: Data loading, cleaning, and preprocessing
- feature_engineer: Feature selection and engineering
- model_trainer: Model training and validation
- evaluator: Model evaluation and metrics
- saver: Model saving and export functionality
- utils: Utility functions and helpers
- main_trainer: Complete training pipeline orchestrator

Structure:
    training/
    ├── __init__.py          # This file - module initialization
    ├── data_processor.py    # Data loading and preprocessing
    ├── feature_engineer.py  # Feature engineering and selection
    ├── model_trainer.py     # Model training and hyperparameter optimization
    ├── evaluator.py         # Model evaluation and metrics
    ├── saver.py            # Model saving and export
    ├── utils.py            # Utility functions and helpers
    └── main_trainer.py     # Main training pipeline

Usage:
    results = run_train(config = {'model_type': 'catboost', 'target_auc': 70.0})

    # Or use individual components
    processor = DataProcessor()
    trainer = ModelTrainer()
"""

# Import with error handling for better debugging

try:
except ImportError as e:
    warnings.warn(f"Could not import DataProcessor: {e}")
    DataProcessor = None

try:
except ImportError as e:
    warnings.warn(f"Could not import FeatureEngineer: {e}")
    FeatureEngineer = None

try:
except ImportError as e:
    warnings.warn(f"Could not import ModelTrainer: {e}")
    ModelTrainer = None

try:
except ImportError as e:
    warnings.warn(f"Could not import ModelEvaluator: {e}")
    ModelEvaluator = None

try:
except ImportError as e:
    warnings.warn(f"Could not import ModelSaver: {e}")
    ModelSaver = None

try:
except ImportError as e:
    warnings.warn(f"Could not import utils: {e}")
    TrainingUtils = None
    ConfigValidator = None

# Main training function (backward compatibility)
try:
except ImportError as e:
    warnings.warn(f"Could not import main_trainer: {e}")

    # Fallback function
    def run_train(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Fallback training function when main_trainer is unavailable"""
        return {
            'success': False, 
            'error': 'Main trainer module not available', 
            'best_auc': 0.0, 
            'target_achieved': False
        }

    MainTrainer = None

# Advanced training capabilities
try:
        AdvancedTrainer, 
        run_ensemble_train, 
        run_hyperopt_train, 
        run_cv_train, 
        run_automl_train
    )
except ImportError as e:
    warnings.warn(f"Could not import advanced_trainer: {e}")
    AdvancedTrainer = None
    run_ensemble_train = None
    run_hyperopt_train = None
    run_cv_train = None
    run_automl_train = None

# Performance monitoring
try:
        PerformanceMonitor, 
        TrainingDashboard, 
        TrainingReporter, 
        TrainingMetrics
    )
except ImportError as e:
    warnings.warn(f"Could not import monitor: {e}")
    PerformanceMonitor = None
    TrainingDashboard = None
    TrainingReporter = None
    TrainingMetrics = None

# Export all available components
__all__ = [
    'DataProcessor', 
    'FeatureEngineer', 
    'ModelTrainer', 
    'ModelEvaluator', 
    'ModelSaver', 
    'TrainingUtils', 
    'ConfigValidator', 
    'MainTrainer', 
    'run_train', 
    'AdvancedTrainer', 
    'run_ensemble_train', 
    'run_hyperopt_train', 
    'run_cv_train', 
    'run_automl_train', 
    'PerformanceMonitor', 
    'TrainingDashboard', 
    'TrainingReporter', 
    'TrainingMetrics'
]

# Version info
__version__ = "2.1.0"
__author__ = "ML Training Team"
__description__ = "Modular ML training pipeline with advanced features"

# Module - level configuration
DEFAULT_CONFIG = {
    'model_type': 'catboost', 
    'target_auc': 70.0, 
    'max_iterations': 5, 
    'cv_folds': 5, 
    'random_state': 42, 
    'verbose': True
}

def get_available_components():
    """Get list of successfully imported components"""
    components = {}
    for component_name in __all__:
        component = globals().get(component_name)
        components[component_name] = component is not None
    return components

def get_module_info():
    """Get detailed module information"""
    return {
        'version': __version__, 
        'author': __author__, 
        'description': __description__, 
        'available_components': get_available_components(), 
        'default_config': DEFAULT_CONFIG
    }