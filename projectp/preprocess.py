import os
import sys
import pandas as pd
from projectp.utils_feature import ensure_super_features_file, get_feature_target_columns

# NOTE: SELECTED_FEATURES จะถูก set จาก pipeline หลัก (global)
SELECTED_FEATURES = None

def _safe_unique(val) -> list[str]:
    # Helper: convert unique() result to list, fallback to str if error
    try:
        return list(val.unique())
    except Exception:
        return [str(x) for x in val]

def run_preprocess():
    """Preprocess pipeline: โหลด feature engineered + target อัตโนมัติ, เลือก feature/target ใหม่, บันทึก preprocessed_super.parquet"""
    fe_super_path = ensure_super_features_file()
    df = pd.read_parquet(fe_super_path)
    feature_cols, target_col = get_feature_target_columns(df)
    print(f'ใช้ feature: {feature_cols}')
    print(f'ใช้ target: {target_col}')
    df_out = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
    # ไม่สร้างคอลัมน์ 'target' ซ้ำถ้ามี target อยู่แล้ว
    if 'target' not in df_out.columns:
        df_out['target'] = df_out[target_col]
    if 'pred_proba' not in df_out.columns:
        df_out['pred_proba'] = 0.5  # default dummy value
    print(f'[DEBUG][preprocess] df_out shape: {df_out.shape}')
    print(f'[DEBUG][preprocess] target unique: {_safe_unique(df_out[target_col])}')
    for col in feature_cols:
        print(f'[DEBUG][preprocess] {col} unique: {_safe_unique(df_out[col])[:5]}')
    if len(_safe_unique(df_out[target_col])) == 1:
        print(f"[STOP][preprocess] Target มีค่าเดียว: {_safe_unique(df_out[target_col])} หยุด pipeline")
        sys.exit(1)
    out_path = os.path.join('output_default', 'preprocessed_super.parquet')
    df_out.to_parquet(out_path)
    print(f'บันทึกไฟล์ preprocessed_super.parquet ด้วย feature/target ใหม่ ({target_col})')
