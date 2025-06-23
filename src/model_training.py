# model_training.py
"""
ฟังก์ชันเกี่ยวกับการ train/export meta model, feature selection, optuna tuning, shap, permutation importance
"""
import os
import time
import logging
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from typing import Dict, List
from joblib import dump as joblib_dump
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report

# Import ML/utility libraries
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = None
    Pool = None

try:
    import shap
except ImportError:
    shap = None

try:
    import optuna
except ImportError:
    optuna = None

# นำเข้า Helper functions
from src.utils.model_utils import predict_with_time_check
from src.data_loader.csv_loader import safe_load_csv_auto
from src.data_loader.m1_loader import load_final_m1_data
from src.utils import get_env_float, load_json_with_comments
from src.utils.leakage import assert_no_overlap
from src.utils.gc_utils import maybe_collect
from src.config import print_gpu_utilization
from src.features import (
    select_top_shap_features,
    check_model_overfit,
    analyze_feature_importance_shap,
    check_feature_noise_shap,
)

# อ่านเวอร์ชันจากไฟล์ VERSION
VERSION_FILE = os.path.join(os.path.dirname(__file__), '..', 'VERSION')
with open(VERSION_FILE, 'r', encoding='utf-8') as vf:
    __version__ = vf.read().strip()

# ค่า default สำหรับ function parameters
DEFAULT_META_CLASSIFIER_PATH = "meta_classifier.pkl"
DEFAULT_SPIKE_MODEL_PATH = "meta_classifier_spike.pkl"
DEFAULT_CLUSTER_MODEL_PATH = "meta_classifier_cluster.pkl"
DEFAULT_MODEL_TO_LINK = "catboost"
DEFAULT_ENABLE_OPTUNA_TUNING = True
DEFAULT_OPTUNA_N_TRIALS = 50
DEFAULT_OPTUNA_CV_SPLITS = 5
DEFAULT_OPTUNA_METRIC = "AUC"
DEFAULT_OPTUNA_DIRECTION = "maximize"
DEFAULT_META_CLASSIFIER_FEATURES = [] # Should be loaded or defined globally
DEFAULT_SHAP_IMPORTANCE_THRESHOLD = 0.01
DEFAULT_PERMUTATION_IMPORTANCE_THRESHOLD = 0.001
DEFAULT_CATBOOST_GPU_RAM_PART = 0.95
DEFAULT_SAMPLE_SIZE = None
DEFAULT_FEATURES_TO_DROP = None
DEFAULT_EARLY_STOPPING_ROUNDS = 200

# --- Meta Model Training Function ---
def train_and_export_meta_model(
    trade_log_path="trade_log_v32_walkforward.csv",
    m1_data_path="final_data_m1_v32_walkforward.csv",
    output_dir=None,
    model_purpose='main',
    trade_log_df_override=None,
    model_type_to_train="catboost",
    link_model_as_default="catboost",
    enable_dynamic_feature_selection=True,
    feature_selection_method='shap',
    shap_importance_threshold=0.01,
    permutation_importance_threshold=0.001,
    prelim_model_params=None,
    enable_optuna_tuning=True,
    optuna_n_trials=50,
    optuna_cv_splits=5,
    optuna_metric="AUC",
    optuna_direction="maximize",
    drift_observer=None,
    catboost_gpu_ram_part=0.95,
    optuna_n_jobs=-1,
    sample_size=None,
    features_to_drop_before_train=None,
    early_stopping_rounds=200,
    enable_threshold_tuning=False,
    fold_index=None,
):
    """
    ฟังก์ชัน Train และ Export Meta Classifier (L1) สำหรับกลยุทธ์ AI เทรดทองคำ
    (รายละเอียด docstring ตามต้นฉบับ)
    """
    # แสดงการใช้งาน GPU
    print_gpu_utilization()

    # --- 0. ตั้งค่าเบื้องต้น ---
    start_time = time.time()
    best_model = None
    best_score = -np.inf
    best_params = None
    study = None

    # กำหนด output directory
    if output_dir is None:
        output_dir = os.path.dirname(trade_log_path)
    os.makedirs(output_dir, exist_ok=True)

    # บันทึกการตั้งค่าเริ่มต้นลงไฟล์
    with open(os.path.join(output_dir, "settings.txt"), "w") as f:
        f.write(f"trade_log_path: {trade_log_path}\n")
        f.write(f"m1_data_path: {m1_data_path}\n")
        f.write(f"output_dir: {output_dir}\n")
        f.write(f"model_purpose: {model_purpose}\n")
        f.write(f"model_type_to_train: {model_type_to_train}\n")
        f.write(f"link_model_as_default: {link_model_as_default}\n")
        f.write(f"enable_dynamic_feature_selection: {enable_dynamic_feature_selection}\n")
        f.write(f"feature_selection_method: {feature_selection_method}\n")
        f.write(f"shap_importance_threshold: {shap_importance_threshold}\n")
        f.write(f"permutation_importance_threshold: {permutation_importance_threshold}\n")
        f.write(f"prelim_model_params: {prelim_model_params}\n")
        f.write(f"enable_optuna_tuning: {enable_optuna_tuning}\n")
        f.write(f"optuna_n_trials: {optuna_n_trials}\n")
        f.write(f"optuna_cv_splits: {optuna_cv_splits}\n")
        f.write(f"optuna_metric: {optuna_metric}\n")
        f.write(f"optuna_direction: {optuna_direction}\n")
        f.write(f"drift_observer: {drift_observer}\n")
        f.write(f"catboost_gpu_ram_part: {catboost_gpu_ram_part}\n")
        f.write(f"optuna_n_jobs: {optuna_n_jobs}\n")
        f.write(f"sample_size: {sample_size}\n")
        f.write(f"features_to_drop_before_train: {features_to_drop_before_train}\n")
        f.write(f"early_stopping_rounds: {early_stopping_rounds}\n")
        f.write(f"enable_threshold_tuning: {enable_threshold_tuning}\n")
        f.write(f"fold_index: {fold_index}\n")

    # --- 1. โหลดข้อมูล ---
    if trade_log_df_override is not None:
        trade_log_df = trade_log_df_override
    else:
        trade_log_df = safe_load_csv_auto(trade_log_path)

    m1_data_df = load_final_m1_data(m1_data_path)

    # แสดงตัวอย่างข้อมูล
    logging.info("ตัวอย่างข้อมูล trade_log_df:")
    logging.info(trade_log_df.head())
    logging.info("ตัวอย่างข้อมูล m1_data_df:")
    logging.info(m1_data_df.head())

    # --- 2. แบ่งข้อมูลสำหรับเทรนและทดสอบ ---
    X = m1_data_df.drop(columns=["signal"])
    y = m1_data_df["signal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 3. Feature Selection (ถ้าเปิดใช้งาน) ---
    if enable_dynamic_feature_selection:
        if feature_selection_method == 'shap' and shap is not None:
            # ใช้ SHAP ในการเลือกฟีเจอร์
            model = CatBoostClassifier(iterations=10, depth=2, learning_rate=1, loss_function="Logloss", verbose=0)
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=0)

            # คำนวณค่า SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)

            # เลือกฟีเจอร์ที่มีค่า SHAP สูงกว่า threshold
            important_features = np.where(np.abs(shap_values).mean(axis=0) > shap_importance_threshold)[0]
            X_train = X_train.iloc[:, important_features]
            X_test = X_test.iloc[:, important_features]

            logging.info(f"เลือกฟีเจอร์ด้วย SHAP เสร็จสิ้น: {X_train.shape[1]} ฟีเจอร์")
        elif feature_selection_method == 'permutation' and permutation_importance_threshold is not None:
            # ใช้ Permutation Importance ในการเลือกฟีเจอร์
            model = CatBoostClassifier(iterations=10, depth=2, learning_rate=1, loss_function="Logloss", verbose=0)
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=0)

            # คำนวณ Permutation Importance
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            importance = result.importances_mean

            # เลือกฟีเจอร์ที่มีค่า Importance สูงกว่า threshold
            important_features = np.where(importance > permutation_importance_threshold)[0]
            X_train = X_train.iloc[:, important_features]
            X_test = X_test.iloc[:, important_features]

            logging.info(f"เลือกฟีเจอร์ด้วย Permutation Importance เสร็จสิ้น: {X_train.shape[1]} ฟีเจอร์")
        else:
            logging.warning("ไม่สามารถเลือกฟีเจอร์ได้ เนื่องจากไม่มีวิธีการที่ถูกต้องหรือไม่มีฟีเจอร์ที่สำคัญพอ")
            important_features = np.arange(X_train.shape[1])
    else:
        important_features = np.arange(X_train.shape[1])

    # --- 4. Train Model ---
    if model_type_to_train == "catboost" and CatBoostClassifier is not None:
        # กำหนดพารามิเตอร์เริ่มต้นสำหรับ CatBoost
        catboost_params = {
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.1,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": 100,
            "early_stopping_rounds": early_stopping_rounds,
        }

        # ปรับพารามิเตอร์ด้วย Optuna (ถ้าเปิดใช้งาน)
        if enable_optuna_tuning and optuna is not None:
            def objective(trial):
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "random_seed": 42,
                    "verbose": 0,
                    "early_stopping_rounds": early_stopping_rounds,
                }

                model = CatBoostClassifier(**params)
                model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)

                # คำนวณ AUC Score
                y_pred = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred)

                return auc_score

            # สร้าง Optuna study
            study = optuna.create_study(direction=optuna_direction)
            study.optimize(objective, n_trials=optuna_n_trials)

            # แสดงผลลัพธ์การปรับพารามิเตอร์
            best_params = study.best_params
            best_score = study.best_value

            logging.info(f"Optuna Tuning เสร็จสิ้น: {best_params} (AUC: {best_score})")
        else:
            best_params = catboost_params

        # เทรนโมเดลด้วยพารามิเตอร์ที่ดีที่สุด
        best_model = CatBoostClassifier(**best_params)
        best_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

        # บันทึกโมเดล
        model_path = os.path.join(output_dir, "catboost_model.cbm")
        best_model.save_model(model_path)

        logging.info(f"โมเดลถูกบันทึกที่: {model_path}")

        # ทดสอบโมเดล
        y_pred = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred)
        logging.info(f"AUC Score บนชุดทดสอบ: {test_auc}")

        # --- 5. บันทึกผลลัพธ์และรายงาน ---
        # บันทึกผลลัพธ์การทดสอบ
        results_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
        })
        results_path = os.path.join(output_dir, "test_results.csv")
        results_df.to_csv(results_path, index=False)

        logging.info(f"ผลลัพธ์การทดสอบถูกบันทึกที่: {results_path}")

        # สร้างรายงานการจำแนกประเภท
        report = classification_report(y_test, np.round(y_pred), output_dict=True)
        report_path = os.path.join(output_dir, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

        logging.info(f"รายงานการจำแนกประเภทถูกบันทึกที่: {report_path}")

    else:
        logging.error(f"ไม่สามารถเทรนโมเดลประเภท '{model_type_to_train}' ได้")
        raise ValueError(f"ไม่สามารถเทรนโมเดลประเภท '{model_type_to_train}' ได้")

    # --- 6. ตรวจสอบการ Overfitting ---
    check_model_overfit(best_model, X_train, y_train, X_test, y_test)

    # --- 7. วิเคราะห์ความสำคัญของฟีเจอร์ ---
    analyze_feature_importance_shap(best_model, X_train, y_train, features=X_train.columns, output_dir=output_dir)

    # --- 8. จัดการ Drift (ถ้ามี) ---
    if drift_observer is not None:
        drift_observer.observe(X_train, y_train, X_test, y_test, model=best_model)

    # --- 9. แสดงผลลัพธ์สุดท้าย ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"ใช้เวลาทั้งหมดในการเทรนและทดสอบโมเดล: {elapsed_time:.2f} วินาที")
    logging.info(f"โมเดลที่ดีที่สุด: {best_model}")
    logging.info(f"คะแนน AUC บนชุดทดสอบ: {test_auc}")

    return best_model, study
