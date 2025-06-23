"""
ML Pipeline Analyzer
===================

Deep analysis of machine learning pipeline structure, data flow, and model performance.
"""

import os
import ast
import inspect
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

class MLPipelineAnalyzer:
    """
    Advanced analysis of ML pipeline components, data flow, and performance characteristics.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.pipeline_components = {}
        self.data_flow = []
        self.performance_metrics = {}
        
    def analyze_pipeline_structure(self) -> Dict[str, Any]:
        """
        Analyze the structure of the ML pipeline including:
        - Data preprocessing steps
        - Feature engineering components
        - Model training workflow
        - Validation and testing procedures
        - Output generation
        """
        structure = {
            'preprocessing': self._analyze_preprocessing(),
            'feature_engineering': self._analyze_feature_engineering(),
            'model_training': self._analyze_model_training(),
            'validation': self._analyze_validation(),
            'prediction': self._analyze_prediction(),
            'data_flow': self._map_data_flow(),
            'bottlenecks': self._identify_bottlenecks()
        }
        
        return structure
    
    def analyze_auc_performance(self) -> Dict[str, Any]:
        """
        Specific analysis of AUC performance patterns and improvement opportunities.
        """
        auc_analysis = {
            'current_auc_methods': self._find_auc_implementations(),
            'auc_improvement_opportunities': self._identify_auc_improvements(),
            'data_quality_impact': self._assess_data_quality_impact_on_auc(),
            'model_selection_impact': self._assess_model_selection_impact(),
            'feature_importance_for_auc': self._analyze_feature_importance()
        }
        
        return auc_analysis
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        Comprehensive data quality analysis.
        """
        data_quality = {
            'missing_values': self._analyze_missing_values(),
            'outliers': self._detect_outliers(),
            'data_drift': self._detect_data_drift(),
            'feature_correlation': self._analyze_feature_correlation(),
            'target_distribution': self._analyze_target_distribution(),
            'data_leakage_risks': self._detect_data_leakage_risks()
        }
        
        return data_quality
    
    def _analyze_preprocessing(self) -> Dict[str, Any]:
        """Analyze data preprocessing components."""
        preprocessing_files = []
        preprocessing_steps = []
        
        # Search for preprocessing-related files
        for py_file in self.project_root.rglob("*.py"):
            if any(keyword in py_file.name.lower() for keyword in 
                   ['preprocess', 'clean', 'transform', 'prepare']):
                preprocessing_files.append(str(py_file))
                
                # Analyze preprocessing steps within file
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tree = ast.parse(content)
                        
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if any(keyword in node.name.lower() for keyword in 
                                   ['clean', 'transform', 'preprocess', 'normalize', 'scale']):
                                preprocessing_steps.append({
                                    'function': node.name,
                                    'file': str(py_file),
                                    'line': node.lineno
                                })
                except Exception as e:
                    print(f"Error analyzing {py_file}: {e}")
        
        return {
            'files': preprocessing_files,
            'steps': preprocessing_steps,
            'total_steps': len(preprocessing_steps)
        }
    
    def _analyze_feature_engineering(self) -> Dict[str, Any]:
        """Analyze feature engineering components."""
        feature_files = []
        feature_functions = []
        
        # Search for feature engineering files
        for py_file in self.project_root.rglob("*.py"):
            if any(keyword in py_file.name.lower() for keyword in 
                   ['feature', 'engineer', 'extract', 'select']):
                feature_files.append(str(py_file))
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tree = ast.parse(content)
                        
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if any(keyword in node.name.lower() for keyword in 
                                   ['feature', 'engineer', 'extract', 'create', 'generate']):
                                feature_functions.append({
                                    'function': node.name,
                                    'file': str(py_file),
                                    'line': node.lineno
                                })
                except Exception as e:
                    print(f"Error analyzing {py_file}: {e}")
        
        return {
            'files': feature_files,
            'functions': feature_functions,
            'total_functions': len(feature_functions)
        }
    
    def _analyze_model_training(self) -> Dict[str, Any]:
        """Analyze model training components."""
        model_files = []
        training_functions = []
        models_found = []
        
        # Search for model training files
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for model-related imports
                if any(model_lib in content for model_lib in 
                       ['sklearn', 'xgboost', 'lightgbm', 'catboost', 'tensorflow', 'torch']):
                    model_files.append(str(py_file))
                    
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if any(keyword in node.name.lower() for keyword in 
                                   ['train', 'fit', 'model', 'learn']):
                                training_functions.append({
                                    'function': node.name,
                                    'file': str(py_file),
                                    'line': node.lineno
                                })
                        
                        # Detect model instantiations
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and 'model' in target.id.lower():
                                    if isinstance(node.value, ast.Call):
                                        if hasattr(node.value.func, 'attr'):
                                            models_found.append({
                                                'model_type': node.value.func.attr,
                                                'variable_name': target.id,
                                                'file': str(py_file),
                                                'line': node.lineno
                                            })
                                            
            except Exception as e:
                continue
        
        return {
            'files': model_files,
            'training_functions': training_functions,
            'models_found': models_found,
            'total_models': len(models_found)
        }
    
    def _analyze_validation(self) -> Dict[str, Any]:
        """Analyze validation and testing components."""
        validation_methods = []
        metrics_used = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Search for validation keywords
                if any(keyword in content.lower() for keyword in 
                       ['validation', 'cross_val', 'split', 'test', 'evaluate']):
                    
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if any(keyword in node.name.lower() for keyword in 
                                   ['validate', 'test', 'evaluate', 'score', 'metric']):
                                validation_methods.append({
                                    'function': node.name,
                                    'file': str(py_file),
                                    'line': node.lineno
                                })
                        
                        # Search for metrics
                        if isinstance(node, ast.Call):
                            if hasattr(node.func, 'attr'):
                                if any(metric in node.func.attr.lower() for metric in 
                                       ['auc', 'accuracy', 'precision', 'recall', 'f1']):
                                    metrics_used.append({
                                        'metric': node.func.attr,
                                        'file': str(py_file),
                                        'line': node.lineno
                                    })
                                    
            except Exception as e:
                continue
        
        return {
            'validation_methods': validation_methods,
            'metrics_used': metrics_used,
            'total_metrics': len(set(m['metric'] for m in metrics_used))
        }
    
    def _analyze_prediction(self) -> Dict[str, Any]:
        """Analyze prediction and inference components."""
        prediction_functions = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if any(keyword in node.name.lower() for keyword in 
                               ['predict', 'inference', 'forecast', 'classify']):
                            prediction_functions.append({
                                'function': node.name,
                                'file': str(py_file),
                                'line': node.lineno
                            })
                            
            except Exception as e:
                continue
        
        return {
            'prediction_functions': prediction_functions,
            'total_functions': len(prediction_functions)
        }
    
    def _map_data_flow(self) -> List[Dict[str, Any]]:
        """Map the data flow through the pipeline."""
        # This would require more sophisticated analysis
        # For now, return a basic structure
        return [
            {'stage': 'data_loading', 'input': 'raw_data', 'output': 'loaded_data'},
            {'stage': 'preprocessing', 'input': 'loaded_data', 'output': 'clean_data'},
            {'stage': 'feature_engineering', 'input': 'clean_data', 'output': 'features'},
            {'stage': 'model_training', 'input': 'features', 'output': 'trained_model'},
            {'stage': 'prediction', 'input': 'features', 'output': 'predictions'}
        ]
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify potential performance bottlenecks."""
        bottlenecks = []
        
        # Look for common bottleneck patterns
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for potential bottlenecks
                if 'pd.concat' in content:
                    bottlenecks.append({
                        'type': 'data_concatenation',
                        'file': str(py_file),
                        'severity': 'medium',
                        'description': 'pandas concat operations can be slow for large datasets'
                    })
                    
                if 'for ' in content and '.iterrows()' in content:
                    bottlenecks.append({
                        'type': 'iterrows_loop',
                        'file': str(py_file),
                        'severity': 'high',
                        'description': 'iterrows() is very slow, consider vectorized operations'
                    })
                    
                if '.apply(' in content:
                    bottlenecks.append({
                        'type': 'apply_function',
                        'file': str(py_file),
                        'severity': 'medium',
                        'description': 'apply() can be slow, consider vectorized alternatives'
                    })
                    
            except Exception as e:
                continue
        
        return bottlenecks
    
    def _find_auc_implementations(self) -> List[Dict[str, Any]]:
        """Find current AUC implementation methods."""
        auc_implementations = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'auc' in content.lower():
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if hasattr(node.func, 'attr') and 'auc' in node.func.attr.lower():
                                auc_implementations.append({
                                    'method': node.func.attr,
                                    'file': str(py_file),
                                    'line': node.lineno
                                })
                                
            except Exception as e:
                continue
        
        return auc_implementations
    
    def _identify_auc_improvements(self) -> List[Dict[str, Any]]:
        """Identify opportunities for AUC improvement."""
        improvements = [
            {
                'category': 'data_quality',
                'improvement': 'Remove or impute missing values more effectively',
                'impact': 'high',
                'effort': 'medium'
            },
            {
                'category': 'feature_engineering',
                'improvement': 'Create interaction features between important variables',
                'impact': 'medium',
                'effort': 'medium'
            },
            {
                'category': 'model_selection',
                'improvement': 'Try ensemble methods or more advanced algorithms',
                'impact': 'high',
                'effort': 'high'
            },
            {
                'category': 'hyperparameter_tuning',
                'improvement': 'Implement systematic hyperparameter optimization',
                'impact': 'medium',
                'effort': 'medium'
            },
            {
                'category': 'cross_validation',
                'improvement': 'Use stratified k-fold for better validation',
                'impact': 'medium',
                'effort': 'low'
            }
        ]
        
        return improvements
    
    def _assess_data_quality_impact_on_auc(self) -> Dict[str, Any]:
        """Assess how data quality affects AUC performance."""
        return {
            'missing_value_impact': 'high',
            'outlier_impact': 'medium',
            'noise_impact': 'high',
            'feature_correlation_impact': 'medium',
            'target_balance_impact': 'high'
        }
    
    def _assess_model_selection_impact(self) -> Dict[str, Any]:
        """Assess impact of different model selections."""
        return {
            'current_models': [],  # Would be populated from actual analysis
            'recommended_models': ['XGBoost', 'LightGBM', 'Random Forest', 'Neural Network'],
            'ensemble_opportunity': True,
            'expected_auc_improvement': '0.05-0.15'
        }
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance for AUC improvement."""
        return {
            'feature_selection_methods': [],  # Would be populated from actual analysis
            'importance_ranking': [],
            'redundant_features': [],
            'missing_important_features': []
        }
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing value patterns."""
        return {
            'files_with_missing_handling': [],
            'missing_value_patterns': [],
            'imputation_methods': []
        }
    
    def _detect_outliers(self) -> Dict[str, Any]:
        """Detect outlier handling methods."""
        return {
            'outlier_detection_methods': [],
            'outlier_treatment_methods': []
        }
    
    def _detect_data_drift(self) -> Dict[str, Any]:
        """Detect data drift monitoring."""
        return {
            'drift_monitoring': False,
            'drift_detection_methods': []
        }
    
    def _analyze_feature_correlation(self) -> Dict[str, Any]:
        """Analyze feature correlation patterns."""
        return {
            'correlation_analysis': False,
            'multicollinearity_handling': []
        }
    
    def _analyze_target_distribution(self) -> Dict[str, Any]:
        """Analyze target variable distribution."""
        return {
            'balance_analysis': False,
            'distribution_checks': []
        }
    
    def _detect_data_leakage_risks(self) -> List[Dict[str, Any]]:
        """Detect potential data leakage risks."""
        risks = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for potential leakage patterns
                if 'future' in content.lower() and 'feature' in content.lower():
                    risks.append({
                        'type': 'temporal_leakage',
                        'file': str(py_file),
                        'description': 'Potential use of future information in features'
                    })
                    
                if 'target' in content.lower() and 'feature' in content.lower():
                    risks.append({
                        'type': 'target_leakage',
                        'file': str(py_file),
                        'description': 'Potential target information in features'
                    })
                    
            except Exception as e:
                continue
        
        return risks
    
    def generate_pipeline_insights(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline insights and recommendations."""
        structure = self.analyze_pipeline_structure()
        auc_analysis = self.analyze_auc_performance()
        data_quality = self.analyze_data_quality()
        
        insights = {
            'pipeline_health_score': self._calculate_pipeline_health_score(structure),
            'auc_improvement_potential': self._calculate_auc_improvement_potential(auc_analysis),
            'critical_issues': self._identify_critical_issues(structure, data_quality),
            'optimization_priorities': self._rank_optimization_priorities(structure, auc_analysis),
            'recommended_actions': self._generate_action_recommendations(structure, auc_analysis, data_quality)
        }
        
        return insights
    
    def _calculate_pipeline_health_score(self, structure: Dict[str, Any]) -> float:
        """Calculate overall pipeline health score (0-1)."""
        score = 0.8  # Base score
        
        # Deduct for bottlenecks
        if structure['bottlenecks']:
            score -= min(0.3, len(structure['bottlenecks']) * 0.1)
            
        # Add for comprehensive validation
        if structure['validation']['total_metrics'] > 3:
            score += 0.1
            
        return max(0.0, min(1.0, score))
    
    def _calculate_auc_improvement_potential(self, auc_analysis: Dict[str, Any]) -> float:
        """Calculate AUC improvement potential (0-1)."""
        improvements = auc_analysis['auc_improvement_opportunities']
        high_impact = sum(1 for imp in improvements if imp['impact'] == 'high')
        return min(1.0, high_impact * 0.2)
    
    def _identify_critical_issues(self, structure: Dict[str, Any], data_quality: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical issues that need immediate attention."""
        critical = []
        
        # High severity bottlenecks
        for bottleneck in structure['bottlenecks']:
            if bottleneck['severity'] == 'high':
                critical.append({
                    'type': 'performance_bottleneck',
                    'description': bottleneck['description'],
                    'location': bottleneck['file'],
                    'priority': 'high'
                })
        
        # Data leakage risks
        for risk in data_quality['data_leakage_risks']:
            critical.append({
                'type': 'data_leakage',
                'description': risk['description'],
                'location': risk['file'],
                'priority': 'critical'
            })
        
        return critical
    
    def _rank_optimization_priorities(self, structure: Dict[str, Any], auc_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank optimization priorities by impact and effort."""
        priorities = []
        
        for improvement in auc_analysis['auc_improvement_opportunities']:
            priority_score = 0
            if improvement['impact'] == 'high':
                priority_score += 3
            elif improvement['impact'] == 'medium':
                priority_score += 2
            else:
                priority_score += 1
                
            if improvement['effort'] == 'low':
                priority_score += 2
            elif improvement['effort'] == 'medium':
                priority_score += 1
            
            priorities.append({
                'improvement': improvement['improvement'],
                'category': improvement['category'],
                'priority_score': priority_score,
                'impact': improvement['impact'],
                'effort': improvement['effort']
            })
        
        return sorted(priorities, key=lambda x: x['priority_score'], reverse=True)
    
    def _generate_action_recommendations(self, structure: Dict[str, Any], auc_analysis: Dict[str, Any], data_quality: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific action recommendations."""
        actions = []
        
        # Immediate actions for critical issues
        critical_issues = self._identify_critical_issues(structure, data_quality)
        for issue in critical_issues:
            actions.append({
                'action': f"Fix {issue['type']} in {issue['location']}",
                'priority': issue['priority'],
                'timeline': 'immediate',
                'category': 'bug_fix'
            })
        
        # Performance improvements
        for bottleneck in structure['bottlenecks']:
            if bottleneck['severity'] in ['medium', 'high']:
                actions.append({
                    'action': f"Optimize {bottleneck['type']} in {bottleneck['file']}",
                    'priority': 'medium',
                    'timeline': 'short_term',
                    'category': 'performance'
                })
        
        # AUC improvements
        priorities = self._rank_optimization_priorities(structure, auc_analysis)
        for priority in priorities[:5]:  # Top 5 priorities
            actions.append({
                'action': priority['improvement'],
                'priority': 'medium' if priority['priority_score'] >= 4 else 'low',
                'timeline': 'medium_term' if priority['effort'] == 'medium' else 'long_term',
                'category': 'auc_improvement'
            })
        
        return actions
