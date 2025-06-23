
# 🔧 GLOBAL_FALLBACK_APPLIED: Comprehensive error handling
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Global exception handler for imports
def safe_import(module_name, fallback_value=None, fallback_message=None):
    """Safely import modules with fallbacks"""
    try:
        parts = module_name.split('.')
        module = __import__(module_name)
        for part in parts[1:]:
            module = getattr(module, part)
        return module
    except ImportError as e:
        if fallback_message:
            print(f"⚠️ {fallback_message}")
        else:
            print(f"⚠️ Failed to import {module_name}, using fallback")
        return fallback_value


# Step: Train/Validate Model (เทพ)
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc as calc_auc
import matplotlib.pyplot as plt
import numpy as np
from projectp.model_guard import check_auc_threshold, check_no_overfitting, check_no_noise, check_no_data_leak
from projectp.pro_log import pro_log
import time
import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_ENABLED = True
except Exception:
    GPU_ENABLED = False
from catboost import CatBoostClassifier
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
try:
    import mlflow
    import mlflow.models
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False
    print("[Warning] mlflow not installed. Experiment tracking will be skipped.")
from contextlib import nullcontext
import warnings
from feature_engineering import log_mutual_info_and_feature_importance
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
try:
    from ft_transformer import FTTransformer
    FTTRANSFORMER_AVAILABLE = True
except ImportError:
    FTTRANSFORMER_AVAILABLE = False
from src.features.ml import build_catboost_model
from src.features.ml_auto_builders import build_xgb_model, build_lgbm_model
from src.utils.resource_auto import print_resource_summary, get_optimal_resource_fraction
from projectp.steps.backtest import load_and_prepare_main_csv
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
import platform
import numpy as np
import sklearn
import catboost
import ta
import optuna
import psutil
import shap
import errno
from projectp.steps.parallel_utils import parallel_fit
from joblib import Parallel, delayed
import threading
from fix_target_values import fix_target_values

console = Console()


warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values>`_ for more details."
)

print_resource_summary()


def log_versions():
    table = Table(title="[bold blue]Environment Versions", show_header=True, header_style="bold magenta")
    table.add_column("Library", style="cyan")
    table.add_column("Version", style="green")
    table.add_row("Python", platform.python_version())
    table.add_row("Pandas", pd.__version__)
    table.add_row("NumPy", np.__version__)
    table.add_row("Scikit-learn", sklearn.__version__)
    table.add_row("CatBoost", catboost.__version__)
    table.add_row("TA", ta.__version__)
    table.add_row("Optuna", optuna.__version__)
    table.add_row("psutil", psutil.__version__)
    table.add_row("SHAP", shap.__version__)
    console.print(table)


def log_resource():
    ram = psutil.virtual_memory()
    ram_str = f"{ram.available/1e9:.1f} / {ram.total/1e9:.1f} GB available"
    gpu_str = "N/A"
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        gpu_str = f"{gpu_mem.free/1e9:.1f} / {gpu_mem.total/1e9:.1f} GB available"
    except Exception:
        pass
    console.print(Panel(f"[bold green]System RAM:[/] {ram_str}\n[bold green]GPU RAM:[/] {gpu_str}", title="[green]Resource Status", border_style="green"))


def log_config_summary(config_dict):
    table = Table(title="[bold blue]Config Summary", show_header=True, header_style="bold blue")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    for k, v in config_dict.items():
        table.add_row(str(k), str(v))
    console.print(table)


def log_next_step():
    console.print(Panel(
        "[bold blue]Next Step Suggestions:[/]\n"
        "- ตรวจสอบผลลัพธ์/กราฟ\n"
        "- ปรับ config/threshold\n"
        "- รันโหมดใหม่ หรือ deploy โมเดล\n"
        "- [Tip] export log, share report, หรือ exit",
        title="[bold green]What to do next?", border_style="bright_blue"
    ))


def log_resource(tag="Resource"):
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    msg = f"RAM: {mem.percent:.1f}% used ({mem.used/1e9:.2f}GB/{mem.total/1e9:.2f}GB), CPU: {cpu:.1f}%"
    if GPU_ENABLED:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            msg += f", GPU: {gpu_mem.used/1e9:.2f}GB/{gpu_mem.total/1e9:.2f}GB"
        except Exception:
            msg += ", GPU: unavailable"
    pro_log(msg, tag=tag)


# --- เทพ: Feature Engineering อัตโนมัติ ---
def auto_feature_engineering(df):
    # --- เทพ: Parallel lag/rolling feature engineering ---
    def create_lag_rolling(lag):
        df[f'close_lag{lag}'] = df['Close'].shift(lag)
        df[f'close_rolling_mean{lag}'] = df['Close'].rolling(lag).mean()
    Parallel(n_jobs=-1)(delayed(create_lag_rolling)(lag) for lag in [1, 3, 5])
    df['high_low_ratio'] = df['High'] / (df['Low'] + 1e-6)
    return df


def run_train(config=None, debug_mode=False):
    from rich.console import Console
    console = Console()
    import subprocess
    import numpy as np
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from src.features.ml_auto_builders import build_xgb_model, build_lgbm_model
    from src.utils.resource_auto import get_optimal_resource_fraction, print_resource_summary
    max_retry = 2
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("ProjectP")
    else:
        pro_log("[Train] mlflow not installed. Skipping experiment tracking", tag="Train")    # --- เทพ: Resource Allocation 80% ---
    ram_gb, gpu_gb = get_optimal_resource_fraction(ram_fraction=0.8, gpu_fraction=0.8)
    print_resource_summary()
    gpu_display = f"{gpu_gb:.2f} GB" if gpu_gb is not None else "N/A"
    console.print(Panel(f"[bold green]Allocated RAM: {ram_gb:.2f} GB | GPU: {gpu_display} (80%)", title="[green]Resource Allocation", border_style="green"))
    # --- เทพ: จองทรัพยากร 80% ---    pro_log(f"[Train] Allocating RAM: {ram_gb}GB, GPU: {gpu_gb if gpu_gb else 'N/A'}GB (80%)", tag="Train")
    # --- เทพ: ปิดการเตือนของ CatBoost ---
    from catboost import CatBoostClassifier, Pool
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="catboost")
    with mlflow.start_run(run_name="train_ensemble_เทพ") if MLFLOW_AVAILABLE else nullcontext():
        for attempt in range(max_retry):
            start_time = time.time()
            pro_log(f"[Train] Attempt {attempt+1}/{max_retry}", tag="Train")
            log_resource("Train")
            with Progress(SpinnerColumn(), BarColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
                task = progress.add_task("[cyan]Loading data...", total=100)
                # ลองใช้ auto features ถ้ามี
                auto_feat_path = os.path.join("output_default", "auto_features.parquet")
                if os.path.exists(auto_feat_path):
                    data_path = auto_feat_path
                    pro_log("[Train] Using auto_features.parquet for training", tag="Train")
                    df = pd.read_parquet(data_path)
                    progress.update(task, advance=30, description="[green]Loaded auto_features.parquet")
                else:
                    # --- Integration อัตโนมัติ: รองรับ CSV Timestamp พ.ศ. ---
                    csv_path = os.path.join("data", "raw", "your_data_file.csv")
                    if os.path.exists(csv_path):
                        df = load_and_prepare_main_csv(csv_path, add_target=True)
                        pro_log(f"[Train] Loaded and prepared CSV: {df.shape}", tag="Train")
                        progress.update(task, advance=30, description="[green]Loaded CSV")
                    else:
                        data_path = os.path.join("output_default", "preprocessed_super.parquet")
                        if not os.path.exists(data_path):
                            pro_log(f"[Train] Feature data not found: {data_path}", level="error", tag="Train")
                            progress.update(task, completed=100, description="[red]Feature data not found")
                            return None
                        df = pd.read_parquet(data_path)
                        progress.update(task, advance=30, description="[green]Loaded preprocessed_super.parquet")
                
                pro_log(f"[Train] Data loaded: {df.shape}", tag="Train")
                
                # PRODUCTION-READY TARGET FIXING - แก้ไขปัญหา "Unknown class label"
                from ultimate_production_fix import UltimateProductionFixer
                fixer = UltimateProductionFixer()
                df = fixer.fix_target_values_ultimate(df, target_col='target')
                
                # Backup: ใช้ fix_target_values ปกติหากยังมีปัญหา
                df = fix_target_values(df, target_col='target')
                
                # Validate target values เพื่อให้แน่ใจ
                unique_targets = df['target'].unique()
                pro_log(f"[Train] Final target values: {sorted(unique_targets)}", tag="Train")
                
                # ต้องมีแค่ 0 และ 1 เท่านั้น
                invalid_targets = [t for t in unique_targets if t not in [0, 1]]
                if invalid_targets:
                    pro_log(f"[Train] WARNING: Invalid target values found: {invalid_targets}", level="warn", tag="Train")
                    # แปลงค่าที่ไม่ถูกต้องให้เป็น binary
                    df['target'] = df['target'].apply(lambda x: 1 if float(x) > 0 else 0)
                    pro_log(f"[Train] Fixed invalid targets to binary: {sorted(df['target'].unique())}", tag="Train")
                pro_log(f"[Train] Data memory usage: {df.memory_usage(deep=True).sum()/1e6:.2f} MB", tag="Train")
                log_resource("Train")
                progress.update(task, advance=20, description="[cyan]Data preprocessing...")
                
                # เลือกฟีเจอร์อัตโนมัติ - แก้ไขปัญหา datetime conversion
                pro_log("[Train] Starting feature selection and data type conversion...", tag="Train")
                
                # กรอง datetime columns และ object columns ให้ครบถ้วน
                datetime_columns = ["target", "Date", "datetime", "Timestamp", "Time", "date", "time", "index"]
                features = []
                
                # สร้าง copy ของ dataframe เพื่อแก้ไข
                df_clean = df.copy()
                
                # Log original data info
                pro_log(f"[Train] Original data types: {df_clean.dtypes.value_counts().to_dict()}", tag="Train")
                
                # ประมวลผลแต่ละ column
                for col in df_clean.columns:
                    try:
                        # ข้าม target และ datetime columns
                        if col.lower() in [dcol.lower() for dcol in datetime_columns]:
                            continue
                        
                        # ตรวจสอบ data type
                        if df_clean[col].dtype == "object":
                            # ลองแปลง string datetime เป็น numeric
                            try:
                                # ตรวจสอบว่าเป็น datetime string หรือไม่
                                non_null_values = df_clean[col].dropna()
                                if len(non_null_values) == 0:
                                    continue
                                    
                                sample_val = str(non_null_values.iloc[0])
                                if any(char in sample_val for char in ['-', ':', '/', ' ']) and len(sample_val) > 8:
                                    # อาจเป็น datetime string - แปลงเป็น timestamp
                                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                                    df_clean[col] = df_clean[col].astype('int64', errors='ignore') // 10**9
                                    features.append(col)
                                    pro_log(f"[Train] Converted datetime column '{col}' to timestamp", tag="Train")
                                else:
                                    # ลองแปลงเป็น numeric
                                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                                    if df_clean[col].dtype in ['float64', 'int64']:
                                        features.append(col)
                                        pro_log(f"[Train] Converted object column '{col}' to numeric", tag="Train")
                            except Exception as e:
                                pro_log(f"[Train] Failed to convert object column '{col}': {e}", level="warn", tag="Train")
                                continue
                        elif df_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                            # เป็น numeric column แล้ว
                            features.append(col)
                    except Exception as e:
                        pro_log(f"[Train] Error processing column '{col}': {e}", level="warn", tag="Train")
                        continue
                
                # อัพเดต df ด้วย cleaned version
                df = df_clean
                target = "target"
                
                pro_log(f"[Train] Selected {len(features)} features after type conversion", tag="Train")
                t0 = time.time()
                train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
                progress.update(task, advance=20, description="[cyan]Train/test split done")
                log_resource("Train")
                X_train, y_train = train_df[features], train_df[target]
                X_test, y_test = test_df[features], test_df[target]
                # Log mutual info and feature importance before training
                log_mutual_info_and_feature_importance(X_train, y_train)
                # --- Export รายชื่อฟีเจอร์ทั้งหมด ---
                features_out_path = os.path.join("output_default", "train_features.txt")
                with open(features_out_path, "w", encoding="utf-8") as f:
                    for c in features:
                        f.write(f"{c}\n")
                pro_log(f"[Train] Exported all features to {features_out_path}", tag="Train")
                # --- Log เฉพาะจำนวนฟีเจอร์และ top N ---
                N = 20
                top_features = features[:N]
                pro_log(f"[Train] Number of features: {len(features)} | Top {N}: {top_features}", tag="Train")
                mlflow.log_param("n_features", len(features))
                mlflow.log_param("top_features", top_features)
                # --- ตัดฟีเจอร์ที่ correlation สูง (ก่อนเทรน) ---
                from projectp.steps.preprocess import remove_highly_correlated_features
                X_train, dropped_corr = remove_highly_correlated_features(X_train, threshold=0.95)
                X_test = X_test.drop(columns=dropped_corr, errors='ignore') if dropped_corr else X_test
                features = [c for c in features if c not in dropped_corr]
                pro_log(f"[Train] Dropped highly correlated features before training: {dropped_corr}", tag="Train")
                # --- เลือกเฉพาะฟีเจอร์สำคัญ (top N MI/SHAP) ---
                # (ใช้ mutual info/feature importance ที่ log_mutual_info_and_feature_importance สร้างไว้)
                # หากมีไฟล์ importance/mi ให้เลือก top N
                mi_path = os.path.join("output_default", "feature_mi.csv")
                if os.path.exists(mi_path):
                    mi_df = pd.read_csv(mi_path)
                    mi_df = mi_df.sort_values(by=mi_df.columns[-1], ascending=False)
                    topN = mi_df.iloc[:N, 0].tolist()
                    features = [f for f in features if f in topN]
                    X_train = X_train[features]
                    X_test = X_test[features]
                    pro_log(f"[Train] Selected top {N} features by MI: {features}", tag="Train")
                # --- Convert int columns to float for MLflow schema robustness ---
                for col in X_train.select_dtypes(include='int').columns:
                    X_train[col] = X_train[col].astype('float')
                    X_test[col] = X_test[col].astype('float')
                progress.update(task, advance=20, description="[cyan]Feature selection complete")
                # Fill NaN in train/test features for robustness
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)                # Class weighting (optional, for imbalance)
                class_weight = None
                if abs(y_train.mean() - 0.5) > 0.05:
                    class_weight = {0: (1-y_train.mean()), 1: y_train.mean()}
                    pro_log(f"[Train] Using class_weight: {class_weight}", tag="Train")
                model_type = config.get('model_type', 'catboost') if config else 'catboost'
                # Model selection
                if model_type == 'catboost':
                    model = build_catboost_model({'iterations': 100, 'verbose': 100, 'class_weights': class_weight}, use_gpu=GPU_ENABLED)
                elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    model = build_lgbm_model({'n_estimators': 100, 'class_weight': 'balanced' if class_weight else None}, use_gpu=GPU_ENABLED)
                elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                    scale_pos_weight = class_weight[1]/class_weight[0] if class_weight else 1
                    model = build_xgb_model({'n_estimators': 100, 'scale_pos_weight': scale_pos_weight}, use_gpu=GPU_ENABLED)
                else:
                    if model_type not in ['catboost', 'lightgbm', 'xgboost']:
                        print(f"[Warning] Unknown model_type: {model_type}, falling back to catboost")
                    else:
                        print(f"[Warning] {model_type} not available, falling back to catboost")
                    model = build_catboost_model({'iterations': 100, 'verbose': 100, 'class_weights': class_weight}, use_gpu=GPU_ENABLED)
                t1 = time.time()
                # กรองเฉพาะคอลัมน์ตัวเลขก่อน fit
                if isinstance(X_train, pd.DataFrame):
                    X_train = X_train.select_dtypes(include=["number"])
                    X_test = X_test.select_dtypes(include=["number"])
                model.fit(X_train, y_train)
                pro_log(f"[Train] Model fit done in {time.time()-t1:.2f}s", tag="Train")
                log_resource("Train")
                t2 = time.time()
                y_train_pred = model.predict_proba(X_train)[:, 1]
                y_test_pred = model.predict_proba(X_test)[:, 1]
                pro_log(f"[Train] Prediction done in {time.time()-t2:.2f}s", tag="Train")
                auc_train = roc_auc_score(y_train, y_train_pred)
                auc_test = roc_auc_score(y_test, y_test_pred)
                acc_test = accuracy_score(y_test, y_test_pred > 0.5)
                prec_test = precision_score(y_test, y_test_pred > 0.5)
                rec_test = recall_score(y_test, y_test_pred > 0.5)
                pro_log(f"[Train] AUC train: {auc_train:.3f}, test: {auc_test:.3f}", tag="Train")
                pro_log(f"[Train] Accuracy: {acc_test:.3f}, Precision: {prec_test:.3f}, Recall: {rec_test:.3f}", tag="Train")
                # MLflow logging
                mlflow.log_param("features", features)
                mlflow.log_metric("auc_train", auc_train)
                mlflow.log_metric("auc_test", auc_test)
                mlflow.log_metric("accuracy_test", acc_test)
                mlflow.log_metric("precision_test", prec_test)
                mlflow.log_metric("recall_test", rec_test)
                # Prepare MLflow signature only (do not send input_example to avoid NaN/int error)
                signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
                mlflow.catboost.log_model(
                    model,
                    "catboost_model",
                    signature=signature
                )
                # --- Save model for serving API (robust path) ---
                model_dir = os.path.abspath(os.path.join(os.getcwd(), "output_default"))
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "catboost_model.pkl")
                try:
                    joblib.dump(model, model_path)
                    pro_log(f"[Train] Saved model to {model_path}", tag="Train")
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)
                except Exception as e:
                    pro_log(f"[Train] ERROR: Failed to save model to {model_path}: {e}", level="error", tag="Train")
                    raise
                # Save features order for prediction compatibility
                features_path = os.path.join("output_default", "train_features.txt")
                os.makedirs(os.path.dirname(features_path), exist_ok=True)
                with open(features_path, "w", encoding="utf-8") as f:
                    for feat in features:
                        f.write(f"{feat}\n")
                pro_log(f"[Train] Saved features order to {features_path}", tag="Train")
                # Save test predictions for threshold step
                test_pred_df = test_df.copy()
                test_pred_df['pred_proba'] = y_test_pred
                test_pred_df[['pred_proba', target]].to_csv(os.path.join('models', 'test_pred.csv'), index=False)
                pro_log(f"[Train] Saved test predictions to models/test_pred.csv", tag="Train")
                metrics = {'auc': auc_test, 'train_auc': auc_train, 'test_auc': auc_test}
                try:
                    check_auc_threshold(metrics)
                    check_no_overfitting(metrics)
                    importances = model.get_feature_importance()
                    feature_importance = {f: imp for f, imp in zip(features, importances)}
                    check_no_noise(feature_importance)
                    # สำเร็จ ไม่ต้อง retry
                    break
                except ValueError as e:
                    pro_log(f"[Train] {e} (auto-improve mode)", level="warn", tag="Train")
                    if attempt == max_retry-1:
                        raise
                    # trigger auto feature generation
                    pro_log("[Train] Triggering auto feature generation...", tag="Train")
                    if debug_mode:
                        subprocess.run(["python", "scripts/auto_feature_generation_tepp.py", "--debug"])
                    else:
                        subprocess.run(["python", "scripts/auto_feature_generation_tepp.py"])
            
            pro_log(f"[Train] Total train step time: {time.time()-start_time:.2f}s", tag="Train")
            log_resource("Train")
            # --- Train TabNet (if available) ---
            tabnet_pred = None
            if TABNET_AVAILABLE:
                tabnet_model = TabNetClassifier(verbose=10)
                tabnet_model.fit(X_train.values, y_train.values, eval_set=[(X_test.values, y_test.values)])
                tabnet_pred = tabnet_model.predict_proba(X_test.values)[:,1]
                mlflow.log_metric("tabnet_auc_test", roc_auc_score(y_test, tabnet_pred))
            # --- Train FTTransformer (if available) ---
            ftt_pred = None
            if FTTRANSFORMER_AVAILABLE:
                ftt_model = FTTransformer()
                ftt_model.fit(X_train.values, y_train.values)
                ftt_pred = ftt_model.predict_proba(X_test.values)[:,1]
                mlflow.log_metric("ftt_auc_test", roc_auc_score(y_test, ftt_pred))
            # --- Ensemble (stacking parallel fit) ---
            preds = [y_test_pred]
            fit_models = []
            fit_X = []
            fit_y = []
            if tabnet_pred is not None:
                fit_models.append(tabnet_model)
                fit_X.append(X_train.values)
                fit_y.append(y_train.values)
                preds.append(tabnet_pred)
            if ftt_pred is not None:
                fit_models.append(ftt_model)
                fit_X.append(X_train.values)
                fit_y.append(y_train.values)
                preds.append(ftt_pred)
            if fit_models:
                # Fit all ensemble models in parallel
                Parallel(n_jobs=-1)(delayed(lambda m, X, y: m.fit(X, y))(m, X, y) for m, X, y in zip(fit_models, fit_X, fit_y))
            if len(preds) > 1:
                meta_X = np.column_stack(preds)
                from sklearn.linear_model import LogisticRegression
                meta_model = LogisticRegression().fit(meta_X, y_test.values)
                ensemble_pred = meta_model.predict_proba(meta_X)[:,1]
                ensemble_auc = roc_auc_score(y_test, ensemble_pred)
                mlflow.log_metric("ensemble_auc_test", ensemble_auc)
                pro_log(f"[Train] Ensemble AUC test: {ensemble_auc:.3f}", tag="Train")
            # --- Save the best model (CatBoost) as the final model ---
            # (You can modify this part to save the ensemble model if needed)
            model_dir = os.path.abspath(os.path.join(os.getcwd(), "output_default"))
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "catboost_model_best.pkl")
            try:
                joblib.dump(model, model_path)
                pro_log(f"[Train] Saved best model to {model_path}", tag="Train")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)
            except Exception as e:
                pro_log(f"[Train] ERROR: Failed to save best model to {model_path}: {e}", level="error", tag="Train")
                raise
            # --- Cross-validation (เทพ parallel) ---
            n_splits = config.get('cv_splits', 5) if config else 5
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            aucs = []
            best_model = None
            best_auc = 0
            best_features = features
            def process_fold(fold_tuple):
                fold, (train_idx, val_idx) = fold_tuple
                X_tr, X_val = df.iloc[train_idx][features], df.iloc[val_idx][features]
                y_tr, y_val = df.iloc[train_idx][target], df.iloc[val_idx][target]
                # --- SHAP Feature Selection ---
                try:
                    from src.features.ml import select_top_shap_features
                    temp_model = build_catboost_model({'iterations': 50, 'verbose': 0}, use_gpu=GPU_ENABLED)
                    temp_model.fit(X_tr, y_tr)
                    import shap
                    explainer = shap.TreeExplainer(temp_model)
                    shap_values = explainer.shap_values(X_tr)
                    selected = select_top_shap_features(shap_values, features, shap_threshold=0.01)
                    if selected and len(selected) > 3:
                        X_tr = X_tr[selected]
                        X_val = X_val[selected]
                        pro_log(f"[Train] SHAP เลือกฟีเจอร์: {selected}", tag="Train")
                except Exception as e:
                    pro_log(f"[Train] SHAP feature selection error: {e}", tag="Train")
                # --- Hyperparameter sweep (เทพ) ---
                param_grid = {
                    'iterations': [100, 200],
                    'learning_rate': [0.03, 0.1],
                    'depth': [4, 6, 8]
                }
                model = build_catboost_model({'verbose': 0}, use_gpu=GPU_ENABLED)
                grid = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
                grid.fit(X_tr, y_tr)
                best = grid.best_estimator_
                y_val_pred = best.predict_proba(X_val)[:, 1]
                auc_val = roc_auc_score(y_val, y_val_pred)
                aucs.append(auc_val)
                if auc_val > best_auc:
                    best_auc = auc_val
                    best_model = best
                    best_features = X_tr.columns.tolist()
                # --- Log ROC/PR curve ---
                fpr, tpr, _ = roc_curve(y_val, y_val_pred)
                precision, recall, _ = precision_recall_curve(y_val, y_val_pred)
                plt.figure()
                plt.plot(fpr, tpr, label=f'ROC fold {fold+1} (AUC={auc_val:.3f})')
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.title('ROC Curve')
                plt.legend()
                plt.savefig(f'output_default/roc_curve_fold{fold+1}.png')
                plt.close()
                plt.figure()
                plt.plot(recall, precision, label=f'PR fold {fold+1}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('PR Curve')
                plt.legend()
                plt.savefig(f'output_default/pr_curve_fold{fold+1}.png')
                plt.close()
                return {'auc': auc_val, 'best': best, 'features': X_tr.columns.tolist()}

            fold_results = Parallel(n_jobs=-1)(delayed(process_fold)((fold, split)) for fold, split in enumerate(skf.split(df[features], df[target])))
            for res in fold_results:
                aucs.append(res['auc'])
                if res['auc'] > best_auc:
                    best_auc = res['auc']
                    best_model = res['best']
                    best_features = res['features']
            pro_log(f"[Train] Cross-validation AUCs: {aucs}", tag="Train")
            pro_log(f"[Train] Mean CV AUC: {np.mean(aucs):.3f}", tag="Train")
            # --- Optimize threshold ---
            y_test_pred = best_model.predict_proba(X_test[best_features])[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            pro_log(f"[Train] Optimal threshold: {optimal_threshold:.3f}", tag="Train")            # --- แจ้งเตือนเมื่อ AUC ต่ำ ---
            if best_auc < 0.7:
                pro_log(f"[Train] WARNING: AUC ต่ำ ({best_auc:.3f})", level="warn", tag="Train")
            # --- Save best model ---
            model_dir = os.path.abspath(os.path.join(os.getcwd(), "output_default"))
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "catboost_model_best_cv.pkl")
            try:
                joblib.dump(best_model, model_path)
                pro_log(f"[Train] Saved best CV model to {model_path}", tag="Train")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)
            except Exception as e:
                pro_log(f"[Train] ERROR: Failed to save best CV model to {model_path}: {e}", level="error", tag="Train")
                raise
            # --- เทพ: Export resource log (RAM/CPU/GPU) ---
            import json
            resource_info = {
                'ram_percent': psutil.virtual_memory().percent,
                'ram_used_gb': psutil.virtual_memory().used / 1e9,
                'ram_total_gb': psutil.virtual_memory().total / 1e9,
                'cpu_percent': psutil.cpu_percent(),
            }
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                def _to_float(val):
                    try:
                        return float(val)
                    except Exception:
                        return 0.0
                resource_info['gpu_used_gb'] = _to_float(getattr(gpu_mem, 'used', 0)) / 1e9 if hasattr(gpu_mem, 'used') else 0.0
                resource_info['gpu_total_gb'] = _to_float(getattr(gpu_mem, 'total', 0)) / 1e9 if hasattr(gpu_mem, 'total') else 0.0
            except Exception:
                resource_info['gpu_used_gb'] = 0.0
                resource_info['gpu_total_gb'] = 0.0
            resource_log_path = os.path.join("output_default", "train_resource_log.json")
            with open(resource_log_path, "w", encoding="utf-8") as f:
                json.dump(resource_info, f, indent=2)
            pro_log(f"[Train] Resource log exported to {resource_log_path}", tag="Train")

            # --- เทพ: Export summary metrics ---
            try:
                summary = {}
                for col in df.select_dtypes(include=[float, int]).columns:
                    summary[col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                    }
                summary_path = os.path.join("output_default", "train_summary_metrics.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                pro_log(f"[Train] Summary metrics exported to {summary_path}", tag="Train")
            except Exception as e:
                pro_log(f"[Train] Summary metrics export error: {e}", level="warn", tag="Train")
