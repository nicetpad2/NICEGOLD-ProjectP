
import os
import pandas as pd
import sys
SELECTED_FEATURES = None

def get_feature_target_columns(df: pd.DataFrame) -> tuple[list[str], str]:
    feature_cols = [c for c in df.columns if c not in ['target_event', 'target_direction', 'Date', 'Time', 'Symbol', 'datetime', 'target', 'pred_proba'] and df[c].dtype != 'O']
    global SELECTED_FEATURES
    if SELECTED_FEATURES:
        feature_cols = [c for c in feature_cols if c in SELECTED_FEATURES]
    # Debug: print columns for troubleshooting
    print(f"[DEBUG] Columns in features file: {list(df.columns)}")
    if 'target_event' in df.columns:
        target_col = 'target_event'
    elif 'target_direction' in df.columns:
        target_col = 'target_direction'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        raise ValueError('ไม่พบ target ที่เหมาะสมในไฟล์ feature engineered')
    return feature_cols, target_col

def ensure_super_features_file() -> str:
    """
    ตรวจสอบและคืน path ของไฟล์ features หลัก (เทพ/robust):
    - ถ้าเจอ output_default/preprocessed_super.parquet ให้ใช้ไฟล์นี้
    - ถ้าเจอ features_main.json ให้ใช้ไฟล์นี้ (feature list)
    - ถ้าไม่เจอ ให้แจ้ง error พร้อม UX/ASCII Art
    """
    parquet_path = os.path.join('output_default', 'preprocessed_super.parquet')
    json_path = 'features_main.json'
    if os.path.exists(parquet_path):
        print(f"[OK] พบไฟล์ features: {parquet_path}")
        return parquet_path
    elif os.path.exists(json_path):
        print(f"[OK] พบไฟล์ features: {json_path}")
        return json_path
    else:
        print("[✘] ไม่พบไฟล์ features หลัก (preprocessed_super.parquet หรือ features_main.json)")
        print("[💡] กรุณารันโหมดเตรียมข้อมูล (Preprocess) หรือสร้างไฟล์ features_main.json ก่อน!")
        sys.exit(1)

def map_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map/rename columns to standard names (เทพ)"""
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["open", "openprice", "open_price", "ราคาเปิด"]:
            col_map[c] = "Open"
        elif cl in ["high", "highprice", "high_price", "ราคาสูงสุด"]:
            col_map[c] = "High"
        elif cl in ["low", "lowprice", "low_price", "ราคาต่ำสุด"]:
            col_map[c] = "Low"
        elif cl in ["close", "closeprice", "close_price", "ราคาปิด"]:
            col_map[c] = "Close"
        elif cl in ["volume", "vol", "ปริมาณ"]:
            col_map[c] = "Volume"
        else:
            col_map[c] = c
    df = df.rename(columns = col_map)
    # Remove duplicate lowercase columns if standard exists
    for col in ["close", "volume", "open", "high", "low"]:
        if col in df.columns and col.capitalize() in df.columns:
            df = df.drop(columns = [col])
    return df

def assert_no_lowercase_columns(df: pd.DataFrame):
    """Assert that no lowercase (open/high/low/close/volume) columns remain"""
    lowercase_cols = [c for c in df.columns if c in ["open", "high", "low", "close", "volume"]]
    assert not lowercase_cols, f"พบคอลัมน์ตัวเล็กที่ไม่ควรเหลือ: {lowercase_cols}"