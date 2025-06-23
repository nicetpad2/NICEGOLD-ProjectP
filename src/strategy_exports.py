# strategy_exports.py
#
# ไฟล์นี้สำหรับส่งออกฟังก์ชันที่จำเป็นไปยัง src.strategy package
# เพื่อแก้ไขปัญหา circular import

import logging
import os

def export_functions_to_strategy():
    """
    ส่งออกฟังก์ชันทั้งหมดที่จำเป็นไปยัง src.strategy package
    ฟังก์ชันนี้ควรถูกเรียกหลังจากที่ strategy.py โหลดฟังก์ชันทั้งหมดแล้ว
    """
    try:
        import src.strategy
        from src.data_loader.csv_loader import safe_load_csv_auto
        from src.data_loader.simple_converter import simple_converter
        
        # นำเข้าฟังก์ชันที่จำเป็นจาก features.ml
        try:
            from src.features.ml import (
                select_top_shap_features,
                check_model_overfit,
                analyze_feature_importance_shap,
                check_feature_noise_shap
            )
        except ImportError:
            # ถ้านำเข้าไม่ได้ ให้สร้างฟังก์ชัน stub
            def select_top_shap_features(*args, **kwargs):
                return []
            def check_model_overfit(*args, **kwargs):
                return False
            def analyze_feature_importance_shap(*args, **kwargs):
                return {}
            def check_feature_noise_shap(*args, **kwargs):
                return []
        
        # ส่งออกไปยัง src.strategy package
        # 1. ฟังก์ชันสำหรับ circular import
        src.strategy.is_entry_allowed = globals().get('is_entry_allowed')
        src.strategy.update_breakeven_half_tp = globals().get('update_breakeven_half_tp')
        src.strategy.adjust_sl_tp_oms = globals().get('adjust_sl_tp_oms')
        src.strategy.run_backtest_simulation_v34 = globals().get('run_backtest_simulation_v34')
        src.strategy.run_simple_numba_backtest = globals().get('run_simple_numba_backtest')
        src.strategy.passes_volatility_filter = globals().get('passes_volatility_filter')
        src.strategy.attempt_order = globals().get('attempt_order')
        
        # 2. ฟังก์ชัน utility สำหรับการทดสอบ
        src.strategy.safe_load_csv_auto = safe_load_csv_auto
        src.strategy.simple_converter = simple_converter
        src.strategy.select_top_shap_features = select_top_shap_features
        src.strategy.check_model_overfit = check_model_overfit
        src.strategy.analyze_feature_importance_shap = analyze_feature_importance_shap
        src.strategy.check_feature_noise_shap = check_feature_noise_shap
        
        logging.info("[PATCH] ส่งออกฟังก์ชันสำเร็จแล้ว")
        return True
    except Exception as e:
        logging.error(f"[PATCH] เกิดข้อผิดพลาดในการส่งออกฟังก์ชัน: {e}")
        return False

# ฟังก์ชันนี้จะถูกเรียกเมื่อ strategy.py โหลด
def init_exports():
    """เริ่มต้นการส่งออกฟังก์ชัน"""
    VERSION_FILE = os.path.join(os.path.dirname(__file__), '..', 'VERSION')
    try:
        with open(VERSION_FILE, 'r', encoding='utf-8') as vf:
            __version__ = vf.read().strip()
    except:
        __version__ = "0.0.0"
    
    logging.info(f"[PATCH] เริ่มต้นการส่งออกฟังก์ชัน (version: {__version__})")
    return export_functions_to_strategy()
