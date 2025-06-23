# strategy_helpers.py
#
# ไฟล์นี้รวบรวมฟังก์ชัน helper ทั้งหมดสำหรับกลยุทธ์การเทรด
# เพื่อป้องกันปัญหา circular import

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# ฟังก์ชันช่วยวิเคราะห์ feature สำหรับฝึกซ้อมโมเดล
def select_top_shap_features(shap_values_val, feature_names, shap_threshold=0.01):
    """
    เลือก features ที่สำคัญที่สุดตามค่า SHAP
    
    Args:
        shap_values_val: ค่า SHAP values
        feature_names: ชื่อ features
        shap_threshold: threshold ขั้นต่ำสำหรับความสำคัญของ feature
        
    Returns:
        list ของ features ที่สำคัญ
    """
    try:
        if shap_values_val is None or feature_names is None:
            return []
        
        # คำนวณความสำคัญของ features โดยใช้ค่าสัมบูรณ์เฉลี่ยของค่า SHAP
        feature_importance = np.abs(shap_values_val).mean(axis=0)
        feature_importance = feature_importance / feature_importance.sum()
        
        # เลือก features ที่มีความสำคัญมากกว่า threshold
        important_indices = np.where(feature_importance > shap_threshold)[0]
        important_features = [feature_names[i] for i in important_indices]
        
        return important_features
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดใน select_top_shap_features: {e}")
        return []

def check_model_overfit(model, X_train, y_train, X_val, y_val, X_test=None, y_test=None, metric="AUC", threshold_pct=15):
    """
    ตรวจสอบว่าโมเดลมีปัญหา overfitting หรือไม่
    
    Args:
        model: โมเดลที่ต้องการตรวจสอบ
        X_train: ข้อมูลฝึกซ้อม features
        y_train: ข้อมูลฝึกซ้อมเป้าหมาย
        X_val: ข้อมูล validation features
        y_val: ข้อมูล validation เป้าหมาย
        X_test: ข้อมูลทดสอบ features (ตัวเลือก)
        y_test: ข้อมูลทดสอบเป้าหมาย (ตัวเลือก)
        metric: ชื่อ metric ที่ใช้ ("accuracy", "AUC", "f1", etc.)
        threshold_pct: เปอร์เซ็นต์ความแตกต่างที่ยอมรับได้
        
    Returns:
        bool: True ถ้า overfit, False ถ้าไม่
    """
    try:
        # คำนวณประสิทธิภาพบนชุดข้อมูลฝึกซ้อม
        y_pred_train = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_train)
        
        # คำนวณประสิทธิภาพบนชุดข้อมูล validation
        y_pred_val = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
        
        # คำนวณค่า metric
        from sklearn import metrics
        
        if metric.upper() == "AUC":
            train_metric = metrics.roc_auc_score(y_train, y_pred_train)
            val_metric = metrics.roc_auc_score(y_val, y_pred_val)
        elif metric.upper() == "ACCURACY":
            y_pred_train_class = np.round(y_pred_train)
            y_pred_val_class = np.round(y_pred_val)
            train_metric = metrics.accuracy_score(y_train, y_pred_train_class)
            val_metric = metrics.accuracy_score(y_val, y_pred_val_class)
        else:
            # ใช้ accuracy เป็นค่าเริ่มต้น
            y_pred_train_class = np.round(y_pred_train)
            y_pred_val_class = np.round(y_pred_val)
            train_metric = metrics.accuracy_score(y_train, y_pred_train_class)
            val_metric = metrics.accuracy_score(y_val, y_pred_val_class)
        
        # คำนวณความแตกต่าง
        diff_pct = abs(train_metric - val_metric) / train_metric * 100
        
        # ตรวจสอบว่า overfit หรือไม่
        is_overfit = diff_pct > threshold_pct and train_metric > val_metric
        
        return is_overfit
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดใน check_model_overfit: {e}")
        return False

def analyze_feature_importance_shap(model, model_type, data_sample, features, output_dir, fold_idx=None):
    """
    วิเคราะห์ความสำคัญของ features ด้วย SHAP
    
    Args:
        model: โมเดลที่ต้องการวิเคราะห์
        model_type: ชนิดของโมเดล (ชื่อ)
        data_sample: ข้อมูลตัวอย่างสำหรับวิเคราะห์ SHAP
        features: รายการชื่อ features
        output_dir: โฟลเดอร์สำหรับบันทึกผลลัพธ์
        fold_idx: ดัชนีของ fold (ตัวเลือก)
    
    Returns:
        dict: ผลการวิเคราะห์ SHAP
    """
    try:
        import os
        import shap
        
        # สร้างโฟลเดอร์สำหรับบันทึกผลลัพธ์
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
        
        # สร้าง SHAP explainer
        if model_type in ['LGBMClassifier', 'LGBMRegressor', 'XGBClassifier', 'XGBRegressor', 'CatBoostClassifier', 'RandomForestClassifier']:
            explainer = shap.TreeExplainer(model)
        else:
            # ใช้ KernelExplainer สำหรับโมเดลอื่นๆ
            explainer = shap.KernelExplainer(model.predict_proba, data_sample.iloc[:100])
        
        # คำนวณค่า SHAP
        shap_values = explainer.shap_values(data_sample)
        
        # ถ้าเป็น multi-class classification, เลือกเฉพาะ class positive (ดัชนี 1)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        
        # บันทึกผลลัพธ์
        result = {
            'shap_values': shap_values,
            'features': features,
            'data_sample': data_sample,
            'output_dir': output_dir
        }
        
        return result
    except Exception as e:
        logging.error(f"   (Error) เกิดข้อผิดพลาดระหว่างการวิเคราะห์ SHAP: {e}", exc_info=True)
        return {'error': str(e)}

def check_feature_noise_shap(shap_values, feature_names, threshold=0.01):
    """
    ตรวจสอบ features ที่มีค่า SHAP ต่ำและอาจเป็นสัญญาณรบกวน
    
    Args:
        shap_values: ค่า SHAP values
        feature_names: ชื่อ features
        threshold: threshold ต่ำสุดสำหรับความสำคัญของ feature
        
    Returns:
        None: แสดงผลลัพธ์การตรวจสอบ
    """
    try:
        # คำนวณความสำคัญของ features โดยใช้ค่าสัมบูรณ์เฉลี่ยของค่า SHAP
        feature_importance = np.abs(shap_values).mean(axis=0)
        total_importance = feature_importance.sum()
        normalized_importance = feature_importance / total_importance
        
        # หา features ที่มีความสำคัญต่ำกว่า threshold
        noisy_indices = np.where(normalized_importance < threshold)[0]
        noisy_features = [feature_names[i] for i in noisy_indices]
        
        # แสดงผลลัพธ์
        if noisy_features:
            noise_percent = len(noisy_features) / len(feature_names) * 100
            logging.info(f"พบ {len(noisy_features)} features ({noise_percent:.1f}%) ที่อาจเป็นสัญญาณรบกวน (SHAP < {threshold})")
            for i, feature in enumerate(noisy_features[:10]):  # แสดงเฉพาะ 10 อันแรก
                importance = normalized_importance[noisy_indices[i]] * 100
                logging.debug(f"  - {feature}: {importance:.4f}%")
            if len(noisy_features) > 10:
                logging.debug(f"  - และอีก {len(noisy_features) - 10} features...")
                
        return noisy_features
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดใน check_feature_noise_shap: {e}")
        return []
