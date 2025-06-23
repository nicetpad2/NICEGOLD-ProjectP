import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
try:
    import cudf
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR
    from cuml.metrics import roc_auc_score as cu_roc_auc_score
    GPU_AVAILABLE = True
    print('[GPU] RAPIDS/cuDF/cuML detected: Using GPU for DataFrame and ML!')
except ImportError:
    GPU_AVAILABLE = False
    print('[CPU] RAPIDS/cuDF/cuML not found: Using CPU fallback (pandas/sklearn)')

# NOTE: ต้องแน่ใจว่า ensure_super_features_file, get_feature_target_columns ถูก import หรือ define stub
from projectp.utils_feature import ensure_super_features_file, get_feature_target_columns
from src.utils.resource_auto import print_resource_summary, get_optimal_resource_fraction
from src.features.ml_auto_builders import build_xgb_model
from src.features.ml import log_target_distribution, balance_classes
from feature_engineering import (
    run_auto_feature_generation, run_feature_interaction, add_domain_and_lagged_features, run_tsfresh_feature_extraction
)
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import argparse
import yaml
import hashlib
import time
from projectp.oms_mm.interface import OMSMMEngine
import pandera as pa
from pandera import Column, DataFrameSchema
import shap

# === Data schema validation (เทพ) ===
def validate_data_schema(df):
    """Validate input data schema using pandera."""
    schema = DataFrameSchema({
        'Close': Column(float),
        'Open': Column(float),
        'High': Column(float),
        'Low': Column(float),
        'Volume': Column(float),
        # ... add more columns as needed ...
    }, coerce=True)
    schema.validate(df)
    return True

# === Anomaly/Drift/Outlier check (เทพ) ===
def check_anomaly_outlier(df):
    # Simple: check for duplicates, missing, outlier (z-score)
    n_dup = df.duplicated().sum()
    n_missing = df.isnull().sum().sum()
    z = ((df.select_dtypes(include='number') - df.mean()) / df.std()).abs()
    n_outlier = (z > 5).sum().sum()
    print(f"[Validation] Duplicates: {n_dup}, Missing: {n_missing}, Outlier(>5σ): {n_outlier}")
    return n_dup, n_missing, n_outlier

# === Lineage/Hash logging (เทพ) ===
def log_lineage_version(path):
    with open(path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    print(f"[Lineage] File: {path}, MD5: {file_hash}")
    return file_hash

# === SHAP explainability (เทพ) ===
def export_shap_report(model, X, out_prefix):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_shap_summary.png')
    plt.close()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_shap_bar.png')
    plt.close()
    print(f"[Explainability] SHAP report exported: {out_prefix}_shap_summary.png, {out_prefix}_shap_bar.png")

# === OMS/MM Integration (เทพ) ===
omsm = OMSMMEngine(initial_capital=100, risk_config=None)
def omsmm_event_logger(event_type, event):
    print(f"[OMS/MM] {event_type}: {event}")
omsm.on_event = omsmm_event_logger

# ฟังก์ชันใหม่: split ข้อมูล train/val/test (stratified, reproducible)
def split_train_val_test(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

# ปรับ train_and_validate_model ให้รองรับ test set และคืนผล test prediction/metrics
def train_validate_test_model():
    """Train, validate, and test model. Return metrics and predictions for all sets."""
    # === เรียกใช้ auto feature generation, interaction, domain, tsfresh ===
    run_auto_feature_generation()
    run_feature_interaction()
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    # --- Normalize column names to lower-case (เทพ) ---
    df.columns = [c.lower() for c in df.columns]
    df = add_domain_and_lagged_features(df)
    try:
        run_tsfresh_feature_extraction(df)
    except Exception as e:
        print('[TSFRESH] ข้ามขั้นตอน:', e)
    feature_cols, target_col = get_feature_target_columns(df)
    # === เลือกเฉพาะ numeric columns เป็น features (เทพ) ===
    drop_cols = ['target', 'Time', 'timestamp', 'Date', 'datetime']
    feature_cols = [c for c in df.columns if c not in drop_cols and (df[c].dtype in [float, int] or np.issubdtype(df[c].dtype, np.number))]
    X = df[feature_cols]
    y = df[target_col]
    # Validation: ไม่มี dtype object ใน X
    assert all([str(dt) != 'object' for dt in X.dtypes]), "พบคอลัมน์ string ใน features กรุณาตรวจสอบ input/preprocess"
    log_target_distribution(y, label='target (all)')
    print(f"[DEBUG][train] shape X: {X.shape}, y: {y.shape}, target unique: {np.unique(y) if not GPU_AVAILABLE else cp.unique(y)}")
    if (cp.unique(y).shape[0] if GPU_AVAILABLE else np.unique(y).shape[0]) == 1:
        print(f"[STOP][train] Target มีค่าเดียว: {cp.unique(y) if GPU_AVAILABLE else np.unique(y)} หยุด pipeline")
        sys.exit(1)
    assert not check_data_leakage(df, target_col), "[STOP] พบ feature leakage ในข้อมูล!"
    # Balance class: oversample minority class (RandomOverSampler) + class_weight
    from sklearn.utils.class_weight import compute_class_weight
    # Oversample (เทพ: GPU/CPU อัตโนมัติ)
    if GPU_AVAILABLE:
        X_res, y_res = oversample_gpu(X, y)
    else:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
    log_target_distribution(y_res, label='target (train_balanced)')
    # Split train/val/test
    if GPU_AVAILABLE:
        # ใช้ cuML/rapids สำหรับ split (หรือแปลงเป็น numpy)
        from cuml.model_selection import train_test_split as cu_train_test_split
        X_temp, X_test, y_temp, y_test = cu_train_test_split(X_res, y_res, test_size=0.15, random_state=42, stratify=y_res)
        val_ratio = 0.15 / (1 - 0.15)
        X_train, X_val, y_train, y_val = cu_train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X_res, y_res, test_size=0.15, val_size=0.15, random_state=42)
    log_target_distribution(y_train, label='target (train_final)')
    log_target_distribution(y_val, label='target (val)')
    log_target_distribution(y_test, label='target (test)')
    # แจ้งเตือนถ้า test set มี class เดียวหรือ imbalance รุนแรง
    unique, counts = np.unique(y_test, return_counts=True) if not GPU_AVAILABLE else cp.unique(y_test, return_counts=True)
    if len(unique) == 1:
        print(f"[WARNING] Test set มี class เดียว: {unique}, ไม่ควรใช้วัด performance!")
    elif min(counts)/max(counts) < 0.05:
        print(f"[WARNING] Test set มี class imbalance รุนแรง: {dict(zip(unique, counts))}")
    # Plot distribution ของ top features (SHAP/MI) ใน train/test
    top_features = ['high_low_spread', 'Volume', 'volatility_5', 'rsi_7', 'macd', 'return_1', 'Close']
    plot_feature_distribution(X_train, X_test, [f for f in top_features if f in X_train.columns])
    # Evaluate baseline DummyClassifier
    evaluate_dummy_classifier(X_train, y_train, X_test, y_test)
    # AutoML: ลองหลายโมเดล/parameter (ใช้ validation set)
    model_grid = []
    if GPU_AVAILABLE:
        model_grid.append(('cuML-RF', cuRF, {'n_estimators': [100, 200], 'max_depth': [5, 10]}))
        model_grid.append(('cuML-LogReg', cuLR, {'C': [0.1, 1.0], 'max_iter': [200]}))
        model_grid.append(('XGBoost-GPU', lambda **params: build_xgb_model(params), {'n_estimators': [100], 'max_depth': [5], 'use_label_encoder': [False], 'eval_metric': ['logloss'], 'tree_method': ['gpu_hist'], 'predictor': ['gpu_predictor']}))
    else:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        model_grid.append(('RandomForest', RandomForestClassifier, {'n_estimators': [100, 200], 'max_depth': [5, 10]}))
        model_grid.append(('LogisticRegression', LogisticRegression, {'C': [0.1, 1.0], 'max_iter': [200]}))
        model_grid.append(('XGBoost', lambda **params: build_xgb_model(params), {'n_estimators': [100], 'max_depth': [5], 'use_label_encoder': [False], 'eval_metric': ['logloss']}))
    from sklearn.model_selection import ParameterGrid
    best_score = -float('inf')
    best_model = None
    best_params = None
    for name, Model, param_grid in model_grid:
        for params in ParameterGrid(param_grid):
            print(f"[AutoML] ลองโมเดล {name} params={params}")
            model = Model(**params) if not (name.startswith('XGBoost')) else Model(**params)
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                y_pred_val = model.predict_proba(X_val)[:,1]
            else:
                y_pred_val = model.predict(X_val)
            auc = cu_roc_auc_score(y_val, y_pred_val) if GPU_AVAILABLE else roc_auc_score(y_val, y_pred_val)
            print(f"[AutoML] {name} val AUC: {auc:.4f}")
            if auc > best_score:
                best_score = auc
                best_model = model
                best_params = params
    # Retrain best model on train+val, test on test set
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    best_model.fit(X_trainval, y_trainval)
    if hasattr(best_model, 'predict_proba'):
        y_pred_test_proba = best_model.predict_proba(X_test)[:,1]
        y_pred_test = (y_pred_test_proba > 0.5).astype(int)
    else:
        y_pred_test = best_model.predict(X_test)
        y_pred_test_proba = y_pred_test
    test_auc = roc_auc_score(y_test, y_pred_test_proba)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)
    test_cm = confusion_matrix(y_test, y_pred_test)
    # Save test predictions/metrics (เพิ่ม index เดิม)
    test_pred_df = pd.DataFrame({
        'row': y_test.index if hasattr(y_test, 'index') else np.arange(len(y_test)),
        'y_true': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': y_pred_test,
        'y_pred_proba': y_pred_test_proba
    })
    test_pred_df.to_csv('output_default/test_predictions.csv', index=False)
    test_pred_df.to_parquet('output_default/test_predictions.parquet')
    import json
    with open('output_default/test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'auc': test_auc, 'report': test_report, 'confusion_matrix': test_cm.tolist()}, f, ensure_ascii=False, indent=2)
    # Plot confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(4,4))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('output_default/test_confusion_matrix.png')
    # Plot pred_proba histogram
    plt.figure(figsize=(6,3))
    sns.histplot(y_pred_test_proba, bins=50, kde=True)
    plt.title('Test Prediction Probability Distribution (pred_proba)')
    plt.xlabel('pred_proba')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig('output_default/test_pred_proba_hist.png')
    print('[plot] บันทึกกราฟ test set ที่ output_default/test_confusion_matrix.png, test_pred_proba_hist.png')
    # === Hybrid Ensemble + Meta-learner tuning ===
    voting, _ = build_hybrid_ensemble(X_train, y_train)
    print('[Ensemble] Training VotingClassifier...')
    voting.fit(X_train, y_train)
    print('[Ensemble] Tuning meta-learner (Optuna/RandomizedSearchCV)...')
    meta_learner = tune_meta_learner(X_train, y_train, voting.estimators)
    stacking = StackingClassifier(estimators=voting.estimators, final_estimator=meta_learner, n_jobs=-1)
    print('[Ensemble] Training StackingClassifier (meta-learner)...')
    stacking.fit(X_train, y_train)
    # Evaluate ensemble
    from sklearn.metrics import roc_auc_score
    y_pred_voting = voting.predict_proba(X_test)[:,1]
    y_pred_stacking = stacking.predict_proba(X_test)[:,1]
    auc_voting = roc_auc_score(y_test, y_pred_voting)
    auc_stacking = roc_auc_score(y_test, y_pred_stacking)
    print(f'[Ensemble] Voting AUC: {auc_voting:.4f}, Stacking (meta) AUC: {auc_stacking:.4f}')
    # Log ensemble results
    import json
    with open('output_default/ensemble_auc.json', 'w') as f:
        json.dump({'voting_auc': auc_voting, 'stacking_auc': auc_stacking}, f, indent=2)
    # วิเคราะห์ผล ensemble
    analyze_ensemble_results(y_test, {'voting': y_pred_voting, 'stacking': y_pred_stacking})
    return {
        'val_auc': best_score,
        'test_auc': test_auc,
        'test_report': test_report,
        'test_cm': test_cm,
        'test_pred_df': test_pred_df
    }

# === Workflow Orchestration (เทพ) ===
def run_pipeline_from_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("[Pipeline] Loaded config:", config)
    t0 = time.time()
    # 1. Train
    print("[Pipeline] Step: train")
    result = train_validate_test_model()
    # 2. Threshold
    from projectp.steps.threshold import run_threshold
    print("[Pipeline] Step: threshold")
    best_threshold = run_threshold(config)
    # 3. Predict
    from projectp.steps.predict import run_predict
    print("[Pipeline] Step: predict")
    run_predict(config)
    # 4. Report
    from projectp.steps.report import run_report
    print("[Pipeline] Step: report")
    run_report(config)
    print(f"[Pipeline] Completed in {time.time()-t0:.2f} sec")

# === CLI Entrypoint (เทพ) ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="เทพ Quant/AI Pipeline")
    parser.add_argument('--pipeline', action='store_true', help='Run full pipeline from config')
    parser.add_argument('--step', type=str, default='train', help='Step to run: train/threshold/predict/report')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()
    if args.pipeline:
        run_pipeline_from_config(args.config)
    else:
        if args.step == 'train':
            train_validate_test_model()
        elif args.step == 'threshold':
            from projectp.steps.threshold import run_threshold
            run_threshold()
        elif args.step == 'predict':
            from projectp.steps.predict import run_predict
            run_predict()
        elif args.step == 'report':
            from projectp.steps.report import run_report
            run_report()
        else:
            print(f"[CLI] Unknown step: {args.step}")
