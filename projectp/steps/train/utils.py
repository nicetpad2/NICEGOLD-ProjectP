"""
Training Utilities Module
========================
Common utilities and helper functions for training pipeline
"""

import os
import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from projectp.pro_log import pro_log

console = Console()

class TrainingUtils:
    """Utility functions for training pipeline"""
    
    def __init__(self):
        self.console = Console()
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate training configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        required_keys = ['model_type', 'data_source']
        
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required config key: {key}")
        
        # Validate model type
        if 'model_type' in config:
            valid_models = ['catboost', 'xgboost', 'lightgbm', 'random_forest']
            if config['model_type'] not in valid_models:
                errors.append(f"Invalid model_type. Must be one of: {valid_models}")
        
        # Validate numeric parameters
        numeric_params = {
            'target_auc': (0, 100),
            'max_iterations': (1, 100),
            'cv_folds': (2, 10)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            if param in config:
                value = config[param]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    errors.append(f"{param} must be between {min_val} and {max_val}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def create_training_id(config: Dict[str, Any]) -> str:
        """
        Create unique training ID based on configuration
        
        Args:
            config: Training configuration
            
        Returns:
            Unique training ID string
        """
        # Create hash from config
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Add timestamp
        timestamp = int(time.time())
        
        return f"train_{config_hash}_{timestamp}"
    
    @staticmethod
    def setup_training_directories(base_path: str, training_id: str) -> Dict[str, str]:
        """
        Setup directory structure for training run
        
        Args:
            base_path: Base directory path
            training_id: Unique training ID
            
        Returns:
            Dictionary of created directory paths
        """
        directories = {
            'base': os.path.join(base_path, training_id),
            'models': os.path.join(base_path, training_id, 'models'),
            'metrics': os.path.join(base_path, training_id, 'metrics'),
            'logs': os.path.join(base_path, training_id, 'logs'),
            'artifacts': os.path.join(base_path, training_id, 'artifacts')
        }
        
        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return directories
    
    @staticmethod
    def save_training_config(config: Dict[str, Any], save_path: str) -> bool:
        """
        Save training configuration to file
        
        Args:
            config: Configuration to save
            save_path: Path to save configuration
            
        Returns:
            Success status
        """
        try:
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            return True
        except Exception as e:
            pro_log.error(f"Failed to save config: {str(e)}")
            return False
    
    @staticmethod
    def load_training_config(config_path: str) -> Optional[Dict[str, Any]]:
        """
        Load training configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary or None if failed
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            pro_log.error(f"Failed to load config: {str(e)}")
            return None
    
    @staticmethod
    def calculate_memory_usage() -> Dict[str, float]:
        """
        Calculate current memory usage
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
        except Exception as e:
            pro_log.warning(f"Could not calculate memory usage: {str(e)}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    
    @staticmethod
    def format_time_duration(seconds: float) -> str:
        """
        Format time duration in human readable format
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers
        
        Args:
            numerator: Numerator value
            denominator: Denominator value
            default: Default value if division by zero
            
        Returns:
            Division result or default
        """
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def get_feature_importance_summary(feature_importance: Dict[str, float], top_n: int = 10) -> Table:
        """
        Create feature importance summary table
        
        Args:
            feature_importance: Dictionary of feature importance scores
            top_n: Number of top features to show
            
        Returns:
            Rich table with feature importance
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        table = Table(title=f"Top {top_n} Feature Importance", show_header=True, header_style="bold blue")
        table.add_column("Rank", style="cyan", width=8)
        table.add_column("Feature", style="magenta", width=30)
        table.add_column("Importance", style="green", width=15)
        table.add_column("Relative %", style="yellow", width=12)
        
        max_importance = max(feature_importance.values()) if feature_importance else 1.0
        
        for rank, (feature, importance) in enumerate(top_features, 1):
            relative_pct = (importance / max_importance) * 100
            table.add_row(
                str(rank),
                feature,
                f"{importance:.4f}",
                f"{relative_pct:.1f}%"
            )
        
        return table
    
    @staticmethod
    def check_data_quality(data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check data quality metrics
        
        Args:
            data: Data to check (DataFrame or dict with DataFrames)
            
        Returns:
            Data quality report
        """
        quality_report = {}
        
        try:
            if isinstance(data, dict):
                # Handle dict of DataFrames
                for key, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        quality_report[key] = TrainingUtils._analyze_dataframe_quality(df)
            elif isinstance(data, pd.DataFrame):
                # Handle single DataFrame
                quality_report['data'] = TrainingUtils._analyze_dataframe_quality(data)
            else:
                quality_report['error'] = f"Unsupported data type: {type(data)}"
                
        except Exception as e:
            quality_report['error'] = f"Quality check failed: {str(e)}"
        
        return quality_report
    
    @staticmethod
    def _analyze_dataframe_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality of a single DataFrame"""
        total_cells = df.shape[0] * df.shape[1]
        
        quality_metrics = {
            'shape': df.shape,
            'total_cells': total_cells,
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 0,
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Check for potential issues
        issues = []
        if quality_metrics['missing_percentage'] > 50:
            issues.append("High missing value percentage (>50%)")
        if quality_metrics['duplicate_rows'] > 0:
            issues.append(f"{quality_metrics['duplicate_rows']} duplicate rows found")
        if df.shape[0] < 100:
            issues.append("Low sample size (<100 rows)")
        
        quality_metrics['issues'] = issues
        quality_metrics['quality_score'] = TrainingUtils._calculate_quality_score(quality_metrics)
        
        return quality_metrics
    
    @staticmethod
    def _calculate_quality_score(metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Penalize missing values
        missing_penalty = min(metrics['missing_percentage'] * 2, 50)
        score -= missing_penalty
        
        # Penalize duplicate rows
        if metrics['duplicate_rows'] > 0:
            duplicate_penalty = min((metrics['duplicate_rows'] / metrics['shape'][0]) * 20, 20)
            score -= duplicate_penalty
        
        # Penalize low sample size
        if metrics['shape'][0] < 100:
            score -= 20
        elif metrics['shape'][0] < 1000:
            score -= 10
        
        return max(score, 0.0)
    
    @staticmethod
    def generate_training_report(results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive training report
        
        Args:
            results: Training results
            save_path: Optional path to save report
            
        Returns:
            Report content as string
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ML TRAINING PIPELINE REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Training summary
        report_lines.append("TRAINING SUMMARY")
        report_lines.append("-"*40)
        report_lines.append(f"Status: {'SUCCESS' if results.get('success', False) else 'FAILED'}")
        report_lines.append(f"Best AUC: {results.get('best_auc', 0):.4f}")
        report_lines.append(f"Target AUC: {results.get('target_auc', 0):.4f}")
        report_lines.append(f"Target Achieved: {'Yes' if results.get('target_achieved', False) else 'No'}")
        report_lines.append(f"Best Iteration: {results.get('best_iteration', 'N/A')}")
        report_lines.append(f"Total Time: {TrainingUtils.format_time_duration(results.get('total_time', 0))}")
        report_lines.append("")
        
        # Error details if any
        if not results.get('success', False) and 'error' in results:
            report_lines.append("ERROR DETAILS")
            report_lines.append("-"*40)
            report_lines.append(f"Error: {results['error']}")
            if 'details' in results:
                report_lines.append(f"Details: {results['details']}")
            report_lines.append("")
        
        # Model details
        if 'results' in results and results['results']:
            model_info = results['results'].get('evaluation', {})
            if model_info:
                report_lines.append("MODEL EVALUATION")
                report_lines.append("-"*40)
                for metric, value in model_info.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"{metric.title()}: {value:.4f}")
                    else:
                        report_lines.append(f"{metric.title()}: {value}")
                report_lines.append("")
        
        report_lines.append("="*80)
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    f.write(report_content)
                pro_log.info(f"Training report saved to: {save_path}")
            except Exception as e:
                pro_log.error(f"Failed to save report: {str(e)}")
        
        return report_content

class ConfigValidator:
    """Enhanced configuration validation"""
    
    DEFAULT_CONFIG = {
        'model_type': 'catboost',
        'target_auc': 70.0,
        'max_iterations': 5,
        'cv_folds': 5,
        'random_state': 42,
        'early_stopping_rounds': 100,
        'verbose': True
    }
    
    @classmethod
    def validate_and_merge(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate config and merge with defaults
        
        Args:
            config: User configuration
            
        Returns:
            Merged and validated configuration
        """
        # Start with defaults
        final_config = cls.DEFAULT_CONFIG.copy()
        
        # Merge user config
        final_config.update(config)
        
        # Validate
        is_valid, errors = TrainingUtils.validate_config(final_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        return final_config

# Export utility functions for easy import
__all__ = [
    'TrainingUtils',
    'ConfigValidator'
]
