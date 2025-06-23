# CLI for ProjectP
import argparse
from projectp.pipeline import run_full_pipeline, run_debug_full_pipeline

# Optimize resource usage: set all BLAS env to use all CPU cores
import os
import multiprocessing
num_cores = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)
print(f"Set all BLAS env to use {num_cores} threads")

# Set environment variable for PySpark/pyarrow timezone warning
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

# Try to enable GPU memory growth for TensorFlow (if installed)
try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow: GPU memory growth set to True")
except Exception as e:
    print("TensorFlow not installed or failed to set GPU memory growth:", e)

# Show GPU info if available (PyTorch)
try:
    import torch
    if torch.cuda.is_available():
        print("PyTorch: GPU available:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
except Exception as e:
    print("PyTorch not installed or no GPU available:", e)

import yaml
from src.features.ml_auto_builders import build_xgb_model, build_lgbm_model

def load_model_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print(f"[Warning] ไม่พบ config.yaml ที่ {config_path} จะใช้ค่า default")
        return None, None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_class = None
    model_params = cfg.get("model_params", {})
    # รองรับ CatBoost, XGBoost, LightGBM, RandomForest
    model_name = cfg.get("model_class", "RandomForestClassifier")
    if model_name.lower().startswith("catboost"):
        try:
            from catboost import CatBoostClassifier
            model_class = CatBoostClassifier
        except ImportError:
            print("[Error] ไม่พบ catboost กรุณาติดตั้งก่อน")
            return None, None
    elif model_name.lower().startswith("xgb"):
        model_class = build_xgb_model
    elif model_name.lower().startswith("lgb") or model_name.lower().startswith("lightgbm"):
        model_class = build_lgbm_model
    else:
        from sklearn.ensemble import RandomForestClassifier
        model_class = RandomForestClassifier
    return model_class, model_params

def load_pipeline_config(config_path="config.yaml"):
    import yaml
    if not os.path.exists(config_path):
        print(f"[Warning] ไม่พบ config.yaml ที่ {config_path} จะใช้ค่า default pipeline")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main_cli():
    parser = argparse.ArgumentParser(description="ProjectP CLI")
    parser.add_argument("--mode", choices=["full_pipeline", "debug_full_pipeline", "preprocess", "realistic_backtest", "realistic_backtest_live"], default=None, help="เลือกโหมด pipeline")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev", help="Environment config")
    args, unknown = parser.parse_known_args()

    print("\n==============================")
    print("  ProjectP Professional Mode")
    print("==============================")
    print(" 1. full_pipeline          - รันทุกขั้นตอนแบบอัตโนมัติ (เทพ ครบระบบ) <== แนะนำ")
    print(" 2. debug_full_pipeline    - ดีบัค: ตรวจสอบทุกจุดของ full_pipeline (log ละเอียด, ไม่หยุดเมื่อ error)")
    print(" 3. preprocess              - เตรียมข้อมูล features (สร้างไฟล์ features_main.json/preprocessed_super.parquet)")
    print(" 4. realistic_backtest      - แบลคเทสเสมือนจริง (เทพ, ใช้ข้อมูลจาก full pipeline, walk-forward)")
    print(" 5. robust_backtest          - แบลคเทสเทพ (walk-forward, parallel, เลือกโมเดลได้)")
    print(" 6. realistic_backtest_live  - แบลคเทสเหมือนเทรดจริง (train เฉพาะอดีต, test อนาคต, save model)")
    print("------------------------------")
    print("[Tip] กด Enter เพื่อเลือกโหมดแนะนำ: full_pipeline")
    print("[Tip] พิมพ์เลข หรือชื่อโหมดก็ได้ เช่น 1 หรือ full_pipeline")

    mode = args.mode
    if not mode:
        while True:
            user_input = input("เลือกโหมด (1-6 หรือชื่อโหมด): ").strip()
            if user_input == "" or user_input == "1" or user_input.lower() == "full_pipeline":
                mode = "full_pipeline"
                break
            elif user_input == "2" or user_input.lower() == "debug_full_pipeline":
                mode = "debug_full_pipeline"
                break
            elif user_input == "3" or user_input.lower() == "preprocess":
                mode = "preprocess"
                break
            elif user_input == "4" or user_input.lower() == "realistic_backtest":
                mode = "realistic_backtest"
                break
            elif user_input == "5" or user_input.lower() == "robust_backtest":
                mode = "robust_backtest"
                break
            elif user_input == "6" or user_input.lower() == "realistic_backtest_live":
                mode = "realistic_backtest_live"
                break
            else:
                print("[Error] ไม่พบโหมด กรุณาเลือกใหม่ (1-6 หรือชื่อโหมด)")

    print(f"[Info] เลือกโหมด: {mode}")
    if mode == "full_pipeline":
        run_full_pipeline()
    elif mode == "debug_full_pipeline":
        run_debug_full_pipeline()
    elif mode == "preprocess":
        from projectp.steps.preprocess import run_preprocess
        run_preprocess()
    elif mode == "realistic_backtest":
        from backtest_engine import run_realistic_backtest
        run_realistic_backtest()
    elif mode == "robust_backtest":
        from backtest_engine import run_robust_backtest
        # --- เลือกโมเดลเทพ ---
        print("\n[Robust Backtest] เลือกโมเดล (1=RandomForest, 2=CatBoost, 3=XGBoost, 4=LightGBM, 5=Custom)")
        print(" 1. RandomForest (default)")
        print(" 2. CatBoost")
        print(" 3. XGBoost")
        print(" 4. LightGBM")
        print(" 5. Custom (import เอง)")
        model_choice = input("เลือกโมเดล (1-5): ").strip()
        model_class = None
        model_params = None
        if model_choice == "2":
            try:
                from catboost import CatBoostClassifier
                model_class = CatBoostClassifier
                model_params = {'thread_count': -1, 'verbose': 0}
            except ImportError:
                print("[Error] ไม่พบ catboost กรุณาติดตั้งก่อน")
                return
        elif model_choice == "3":
            model_class = build_xgb_model
            model_params = {'n_jobs': -1, 'verbosity': 0}
        elif model_choice == "4":
            model_class = build_lgbm_model
            model_params = {'n_jobs': -1, 'verbose': -1}
        elif model_choice == "5":
            print("[Custom] โปรดแก้ไขโค้ดเพื่อ import และกำหนด model_class/model_params เอง")
            return
        # Default: RandomForest
        if model_class is None:
            from sklearn.ensemble import RandomForestClassifier
            model_class = RandomForestClassifier
            model_params = {'n_jobs': -1, 'random_state': 42}
        run_robust_backtest(model_class=model_class, model_params=model_params)
    elif mode == "realistic_backtest_live":
        from backtest_engine import run_realistic_backtest_live_model
        # --- โหลด config อัตโนมัติ ---
        model_class, model_params = load_model_config()
        run_realistic_backtest_live_model(model_class=model_class, model_params=model_params)

# ในแต่ละจุดของ pipeline (เช่น run_full_pipeline, run_realistic_backtest_live_model) ให้ส่ง config ที่อ่านจาก config.yaml
# ตัวอย่างใน main_cli หรือ pipeline:
# config = load_pipeline_config()
# run_full_pipeline(config=config)
# หรือ
# run_realistic_backtest_live_model(config=config, model_class=model_class, model_params=model_params)
