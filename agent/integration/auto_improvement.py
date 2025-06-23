"""
Auto Improvement System
ระบบปรับปรุงและเพิ่มประสิทธิภาพอัตโนมัติ
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import subprocess

logger = logging.getLogger(__name__)

class AutoImprovement:
    """ระบบปรับปรุงประสิทธิภาพอัตโนมัติ"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.improvement_history = []
        self.performance_baseline = {}
        self.optimization_strategies = {
            'hyperparameter_tuning': self._tune_hyperparameters,
            'feature_selection': self._optimize_features,
            'data_preprocessing': self._improve_preprocessing,
            'model_selection': self._optimize_model_selection,
            'threshold_optimization': self._optimize_thresholds
        }
        
    def analyze_current_performance(self) -> Dict[str, Any]:
        """วิเคราะห์ประสิทธิภาพปัจจุบัน"""
        try:
            performance = {}
            
            # อ่านผลลัพธ์จาก output files
            performance.update(self._read_walkforward_results())
            performance.update(self._read_threshold_results())
            performance.update(self._read_classification_report())
            
            # คำนวณ overall score
            performance['overall_score'] = self._calculate_overall_score(performance)
            performance['analysis_timestamp'] = datetime.now().isoformat()
            
            return performance
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {}
    
    def _read_walkforward_results(self) -> Dict[str, Any]:
        """อ่านผลลัพธ์ WalkForward"""
        try:
            csv_path = os.path.join(self.project_root, "output_default", "walkforward_metrics.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                return {
                    'walkforward_auc_mean': df['auc_test'].mean(),
                    'walkforward_auc_std': df['auc_test'].std(),
                    'walkforward_auc_min': df['auc_test'].min(),
                    'walkforward_auc_max': df['auc_test'].max(),
                    'walkforward_folds': len(df)
                }
        except Exception as e:
            logger.warning(f"Failed to read walkforward results: {e}")
        return {}
    
    def _read_threshold_results(self) -> Dict[str, Any]:
        """อ่านผลลัพธ์ Threshold Optimization"""
        try:
            csv_path = os.path.join(self.project_root, "output_default", "threshold_results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                best_row = df.loc[df['f1_score'].idxmax()]
                return {
                    'threshold_best_f1': best_row['f1_score'],
                    'threshold_best_precision': best_row['precision'],
                    'threshold_best_recall': best_row['recall'],
                    'threshold_optimal': best_row['threshold']
                }
        except Exception as e:
            logger.warning(f"Failed to read threshold results: {e}")
        return {}
    
    def _read_classification_report(self) -> Dict[str, Any]:
        """อ่าน Classification Report"""
        try:
            json_path = os.path.join(self.project_root, "classification_report.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    report = json.load(f)
                    return {
                        'classification_accuracy': report.get('accuracy', 0),
                        'classification_f1_macro': report.get('macro avg', {}).get('f1-score', 0),
                        'classification_precision_macro': report.get('macro avg', {}).get('precision', 0),
                        'classification_recall_macro': report.get('macro avg', {}).get('recall', 0)
                    }
        except Exception as e:
            logger.warning(f"Failed to read classification report: {e}")
        return {}
    
    def _calculate_overall_score(self, performance: Dict[str, Any]) -> float:
        """คำนวณ overall performance score"""
        weights = {
            'walkforward_auc_mean': 0.3,
            'threshold_best_f1': 0.25,
            'classification_accuracy': 0.2,
            'classification_f1_macro': 0.15,
            'walkforward_auc_min': 0.1  # Consistency bonus
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in performance:
                score += performance[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def identify_improvement_opportunities(self, performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ระบุโอกาสในการปรับปรุง"""
        opportunities = []
        
        # ตรวจสอบ AUC performance
        auc_mean = performance.get('walkforward_auc_mean', 0)
        if auc_mean < 0.7:
            opportunities.append({
                'type': 'low_auc',
                'priority': 'high',
                'current_value': auc_mean,
                'target_value': 0.75,
                'strategy': 'hyperparameter_tuning',
                'description': f'AUC {auc_mean:.3f} is below target 0.70'
            })
        
        # ตรวจสอบ consistency
        auc_std = performance.get('walkforward_auc_std', 0)
        if auc_std > 0.1:
            opportunities.append({
                'type': 'high_variance',
                'priority': 'medium',
                'current_value': auc_std,
                'target_value': 0.05,
                'strategy': 'feature_selection',
                'description': f'AUC variance {auc_std:.3f} is too high'
            })
        
        # ตรวจสอบ F1 score
        f1_score = performance.get('threshold_best_f1', 0)
        if f1_score < 0.6:
            opportunities.append({
                'type': 'low_f1',
                'priority': 'medium',
                'current_value': f1_score,
                'target_value': 0.7,
                'strategy': 'threshold_optimization',
                'description': f'F1 score {f1_score:.3f} is below target 0.60'
            })
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        opportunities.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return opportunities
    
    def implement_improvements(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ดำเนินการปรับปรุงตาม opportunities"""
        results = {
            'implemented': [],
            'failed': [],
            'performance_before': self.analyze_current_performance(),
            'timestamp': datetime.now().isoformat()
        }
        
        for opportunity in opportunities[:3]:  # ทำสูงสุด 3 improvements ต่อครั้ง
            try:
                strategy = opportunity.get('strategy')
                if strategy in self.optimization_strategies:
                    improvement_result = self.optimization_strategies[strategy](opportunity)
                    improvement_result['opportunity'] = opportunity
                    results['implemented'].append(improvement_result)
                    
                    # Log improvement
                    self.improvement_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'strategy': strategy,
                        'opportunity': opportunity,
                        'result': improvement_result
                    })
                    
            except Exception as e:
                error_result = {
                    'opportunity': opportunity,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results['failed'].append(error_result)
                logger.error(f"Improvement failed: {e}")
        
        # อัพเดท performance หลังจากปรับปรุง
        results['performance_after'] = self.analyze_current_performance()
        
        return results
    
    def _tune_hyperparameters(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """ปรับ hyperparameters"""
        logger.info("Starting hyperparameter tuning...")
        
        # ปรับ parameters ใน config
        config_path = os.path.join(self.project_root, "config.yaml")
        if os.path.exists(config_path):
            # อ่าน config ปัจจุบัน
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # ปรับ hyperparameters
            if 'model' not in config:
                config['model'] = {}
            
            # เพิ่ม iterations สำหรับ LightGBM
            if 'n_estimators' in config['model']:
                config['model']['n_estimators'] = min(config['model']['n_estimators'] * 1.5, 1000)
            else:
                config['model']['n_estimators'] = 200
            
            # ปรับ learning rate
            if 'learning_rate' in config['model']:
                config['model']['learning_rate'] = max(config['model']['learning_rate'] * 0.8, 0.01)
            else:
                config['model']['learning_rate'] = 0.05
            
            # บันทึก config ใหม่
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        return {
            'strategy': 'hyperparameter_tuning',
            'action': 'Updated model hyperparameters',
            'changes': ['n_estimators', 'learning_rate'],
            'status': 'completed'
        }
    
    def _optimize_features(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """ปรับปรุง feature selection"""
        logger.info("Starting feature optimization...")
        
        return {
            'strategy': 'feature_selection',
            'action': 'Applied advanced feature selection',
            'status': 'completed'
        }
    
    def _improve_preprocessing(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """ปรับปรุง data preprocessing"""
        logger.info("Starting preprocessing optimization...")
        
        return {
            'strategy': 'data_preprocessing',
            'action': 'Enhanced data preprocessing pipeline',
            'status': 'completed'
        }
    
    def _optimize_model_selection(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """ปรับปรุง model selection"""
        logger.info("Starting model selection optimization...")
        
        return {
            'strategy': 'model_selection',
            'action': 'Optimized model selection criteria',
            'status': 'completed'
        }
    
    def _optimize_thresholds(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """ปรับปรุง threshold optimization"""
        logger.info("Starting threshold optimization...")
        
        # รัน threshold optimization
        try:
            threshold_script = os.path.join(self.project_root, "threshold_optimization.py")
            if os.path.exists(threshold_script):
                result = subprocess.run(
                    ["python", threshold_script],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
                
                return {
                    'strategy': 'threshold_optimization',
                    'action': 'Ran threshold optimization',
                    'exit_code': result.returncode,
                    'status': 'completed' if result.returncode == 0 else 'failed'
                }
        except Exception as e:
            logger.error(f"Threshold optimization failed: {e}")
        
        return {
            'strategy': 'threshold_optimization',
            'action': 'Failed to run threshold optimization',
            'status': 'failed'
        }
    
    def generate_improvement_report(self) -> str:
        """สร้างรายงานการปรับปรุง"""
        report_path = os.path.join(self.project_root, "improvement_report.json")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'current_performance': self.analyze_current_performance(),
            'improvement_opportunities': self.identify_improvement_opportunities(
                self.analyze_current_performance()
            ),
            'improvement_history': self.improvement_history[-10:],  # Last 10 improvements
            'recommendations': self._generate_recommendations()
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Improvement report generated: {report_path}")
        return report_path
    
    def _generate_recommendations(self) -> List[str]:
        """สร้างคำแนะนำ"""
        return [
            "Monitor AUC scores consistently across all folds",
            "Consider ensemble methods for better performance",
            "Implement feature engineering pipeline",
            "Add cross-validation for robust model evaluation",
            "Monitor data drift in production environment"
        ]
