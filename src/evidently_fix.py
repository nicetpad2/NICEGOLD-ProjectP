"""
Comprehensive Evidently Compatibility Fix
แก้ไขปัญหา Evidently ทุกเวอร์ชัน รวมถึง 0.4.30
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Global flag
EVIDENTLY_AVAILABLE = False

class ComprehensiveFallbackValueDrift:
    """Fallback ที่ครอบคลุมสำหรับ ValueDrift"""
    
    def __init__(self, column_name: str = "target", stattest: str = "ks", **kwargs):
        self.column_name = column_name
        self.stattest = stattest
        self.kwargs = kwargs
        logger.info(f"🔄 Using comprehensive fallback ValueDrift for column: {column_name}")
    
    def calculate(self, reference_data, current_data):
        """คำนวณ drift โดยใช้วิธีทางสถิติ"""
        try:
            if isinstance(reference_data, pd.DataFrame) and isinstance(current_data, pd.DataFrame):
                return self._statistical_drift_test(reference_data, current_data)
            else:
                return self._simple_drift_test(reference_data, current_data)
        except Exception as e:
            logger.warning(f"Drift calculation failed: {e}")
            return {
                'drift_score': 0.0,
                'drift_detected': False,
                'p_value': 1.0,
                'method': 'fallback_error',
                'error': str(e)
            }
    
    def _statistical_drift_test(self, ref_data, curr_data):
        """ทดสอบ drift ด้วยวิธีทางสถิติ"""
        try:
            import scipy.stats as stats
            
            # เลือกคอลัมน์ที่จะทดสอบ
            if self.column_name in ref_data.columns and self.column_name in curr_data.columns:
                ref_values = ref_data[self.column_name].dropna()
                curr_values = curr_data[self.column_name].dropna()
            else:
                # ใช้คอลัมน์ตัวเลขทั้งหมด
                numeric_cols = ref_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    ref_values = ref_data[numeric_cols].mean(axis=1).dropna()
                    curr_values = curr_data[numeric_cols].mean(axis=1).dropna()
                else:
                    return self._simple_statistical_comparison(ref_data, curr_data)
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                return {'drift_detected': False, 'method': 'no_data'}
            
            # ทดสอบ Kolmogorov-Smirnov
            if self.stattest == 'ks' or self.stattest == 'auto':
                ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
                drift_detected = p_value < 0.05
                
                return {
                    'drift_score': float(ks_stat),
                    'drift_detected': bool(drift_detected),
                    'p_value': float(p_value),
                    'stattest': 'ks',
                    'method': 'statistical_ks_test'
                }
            
            # ทดสอบ Mann-Whitney U
            elif self.stattest == 'mannw':
                stat, p_value = stats.mannwhitneyu(ref_values, curr_values, alternative='two-sided')
                drift_detected = p_value < 0.05
                
                return {
                    'drift_score': float(stat),
                    'drift_detected': bool(drift_detected),
                    'p_value': float(p_value),
                    'stattest': 'mannw',
                    'method': 'statistical_mannw_test'
                }
            
            # ทดสอบ Wasserstein
            else:
                wasserstein_dist = stats.wasserstein_distance(ref_values, curr_values)
                # กำหนด threshold แบบ adaptive
                ref_std = ref_values.std()
                threshold = ref_std * 0.1 if ref_std > 0 else 0.1
                drift_detected = wasserstein_dist > threshold
                
                return {
                    'drift_score': float(wasserstein_dist),
                    'drift_detected': bool(drift_detected),
                    'threshold': float(threshold),
                    'stattest': 'wasserstein',
                    'method': 'statistical_wasserstein'
                }
                
        except ImportError:
            logger.warning("scipy not available, using simple comparison")
            return self._simple_statistical_comparison(ref_data, curr_data)
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return self._simple_statistical_comparison(ref_data, curr_data)
    
    def _simple_statistical_comparison(self, ref_data, curr_data):
        """เปรียบเทียบแบบง่าย"""
        try:
            # คำนวณสถิติพื้นฐาน
            ref_mean = ref_data.mean().mean() if hasattr(ref_data, 'mean') else 0
            curr_mean = curr_data.mean().mean() if hasattr(curr_data, 'mean') else 0
            
            ref_std = ref_data.std().mean() if hasattr(ref_data, 'std') else 1
            
            # คำนวณ drift score
            drift_score = abs(ref_mean - curr_mean) / max(ref_std, 1e-8)
            drift_detected = drift_score > 2.0  # 2 standard deviations
            
            return {
                'drift_score': float(drift_score),
                'drift_detected': bool(drift_detected),
                'ref_mean': float(ref_mean),
                'curr_mean': float(curr_mean),
                'method': 'simple_statistical'
            }
        except Exception as e:
            return {
                'drift_score': 0.0,
                'drift_detected': False,
                'method': 'fallback_dummy',
                'error': str(e)
            }
    
    def _simple_drift_test(self, ref_data, curr_data):
        """ทดสอบ drift แบบง่ายสำหรับข้อมูลที่ไม่ใช่ DataFrame"""
        try:
            ref_array = np.array(ref_data).flatten()
            curr_array = np.array(curr_data).flatten()
            
            if len(ref_array) == 0 or len(curr_array) == 0:
                return {'drift_detected': False, 'method': 'no_data'}
            
            ref_mean = np.mean(ref_array)
            curr_mean = np.mean(curr_array)
            ref_std = np.std(ref_array)
            
            drift_score = abs(ref_mean - curr_mean) / max(ref_std, 1e-8)
            drift_detected = drift_score > 1.5
            
            return {
                'drift_score': float(drift_score),
                'drift_detected': bool(drift_detected),
                'method': 'simple_array'
            }
        except Exception as e:
            return {
                'drift_score': 0.0,
                'drift_detected': False,
                'method': 'fallback_error',
                'error': str(e)
            }

class ComprehensiveFallbackDataDrift:
    """Fallback ที่ครอบคลุมสำหรับ DataDrift"""
    
    def __init__(self, columns=None, **kwargs):
        self.columns = columns
        self.kwargs = kwargs
        logger.info("🔄 Using comprehensive fallback DataDrift")
    
    def calculate(self, reference_data, current_data):
        """คำนวณ data drift สำหรับหลายคอลัมน์"""
        try:
            if not isinstance(reference_data, pd.DataFrame) or not isinstance(current_data, pd.DataFrame):
                return {'drift_detected': False, 'method': 'not_dataframe'}
            
            columns_to_check = self.columns or reference_data.select_dtypes(include=[np.number]).columns
            drift_results = {}
            drifted_columns = []
            
            for col in columns_to_check:
                if col in reference_data.columns and col in current_data.columns:
                    value_drift = ComprehensiveFallbackValueDrift(column_name=col)
                    result = value_drift.calculate(reference_data, current_data)
                    drift_results[col] = result
                    
                    if result.get('drift_detected', False):
                        drifted_columns.append(col)
            
            total_columns = len(columns_to_check)
            drifted_count = len(drifted_columns)
            drift_share = drifted_count / total_columns if total_columns > 0 else 0
            
            return {
                'drift_detected': drift_share > 0.3,  # 30% threshold
                'number_of_drifted_columns': drifted_count,
                'share_of_drifted_columns': drift_share,
                'drifted_columns': drifted_columns,
                'drift_by_columns': drift_results,
                'method': 'comprehensive_fallback'
            }
            
        except Exception as e:
            logger.warning(f"DataDrift calculation failed: {e}")
            return {
                'drift_detected': False,
                'method': 'fallback_error',
                'error': str(e)
            }

def detect_and_import_evidently():
    """ตรวจสอบและ import Evidently"""
    global EVIDENTLY_AVAILABLE
    
    # ลิสต์ของการ import ที่อาจเป็นไปได้
    import_attempts = [
        # Evidently v0.4.30+
        ("evidently.metrics", "ValueDrift"),
        ("evidently.metrics", "DataDrift"),
        # Evidently v0.3.x
        ("evidently.metrics.data_drift.value_drift_metric", "ValueDrift"),
        # Evidently v0.2.x
        ("evidently.analyzers", "DataDriftAnalyzer"),
        # Evidently v0.1.x
        ("evidently.dashboard", "Dashboard"),
    ]
    
    evidently_classes = {}
    
    for module_name, class_name in import_attempts:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                evidently_classes[class_name] = getattr(module, class_name)
                logger.info(f"✅ Found {class_name} in {module_name}")
        except ImportError:
            continue
    
    # ตรวจสอบว่าพบ ValueDrift หรือไม่
    if 'ValueDrift' in evidently_classes:
        EVIDENTLY_AVAILABLE = True
        logger.info("✅ Evidently ValueDrift available")
        return evidently_classes['ValueDrift'], evidently_classes.get('DataDrift')
    else:
        logger.warning("⚠️ Evidently ValueDrift not found, using comprehensive fallback")
        return ComprehensiveFallbackValueDrift, ComprehensiveFallbackDataDrift

# Initialize Evidently
ValueDrift, DataDrift = detect_and_import_evidently()

# Make them available globally
import builtins
builtins.EvidentlyValueDrift = ValueDrift
builtins.EvidentlyDataDrift = DataDrift
builtins.EVIDENTLY_AVAILABLE = EVIDENTLY_AVAILABLE

logger.info("✅ Comprehensive Evidently compatibility ready")
