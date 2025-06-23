# auto_feature_generation_tepp.py
"""
เทพ Auto Feature Generation สำหรับ Time Series/Trading
- ใช้ Featuretools + tsfresh + technical indicators + interaction + adaptive logic
- รองรับข้อมูล OHLCV, ปรับตัวได้กับทุก dataset ที่มี Date/Time/Price/Volume
- ผลลัพธ์: output_default/auto_features.parquet, summary, csv, plot
"""
import warnings
warnings.filterwarnings("ignore", message="Only one dataframe in entityset, changing max_depth to 1*")

def main(debug_mode=False):
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    try:
        import featuretools as ft
    except ImportError:
        ft = None
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
    except ImportError:
        extract_features = None
        MinimalFCParameters = None
        print("[TIP] ติดตั้ง tsfresh ด้วย pip install tsfresh เพื่อใช้ auto feature extraction เพิ่มเติม")
    try:
        import ta
    except ImportError:
        ta = None
    import psutil

    # Robust path setup
    BASE_DIR = os.path.abspath(os.getcwd())
    OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "output_default"))
    os.makedirs(OUT_DIR, exist_ok=True)
    DATA_PATH = os.path.join(OUT_DIR, "preprocessed_super.parquet")
    OUT_PATH = os.path.join(OUT_DIR, "auto_features.parquet")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data not found: {DATA_PATH}")
        exit(1)
    df = pd.read_parquet(DATA_PATH)

    print('[DEBUG] Initial data shape:', df.shape)
    print('[DEBUG] Data types:', df.dtypes)
    print('[DEBUG] Data head:', df.head())
    print('[DEBUG] RAM used:', psutil.virtual_memory().percent, '%')

    # --- Robust Date/Time column selection for tsfresh ---
    # Try to find a real datetime column (case-insensitive)
    datetime_candidates = [c for c in df.columns if c.lower() in ["date", "datetime", "timestamp", "time"]]
    date_col = None
    for c in ["date", "datetime", "timestamp", "time"]:
        for col in df.columns:
            if col.lower() == c:
                date_col = col
                break
        if date_col:
            break
    if date_col:
        df["Date"] = pd.to_datetime(df[date_col])
    else:
        print('[INFO] ไม่พบ column Date/Datetime/Time ในข้อมูล สร้าง Date อัตโนมัติสำหรับ tsfresh')
        df["Date"] = pd.date_range(start='2020-01-01', periods=len(df), freq='min')

    # --- 1. Basic technical features ---
    def add_technical_features(df):
        if ta is None:
            print("[WARN] ta-lib/ta not installed, skipping technical indicators.")
            return df
        df = df.copy()
        for win in [5, 10, 20, 50]:
            df[f"ma{win}"] = ta.trend.sma_indicator(df["Close"], window=win)
            df[f"std{win}"] = df["Close"].rolling(win).std()
            df[f"roc{win}"] = ta.momentum.roc(df["Close"], window=win)
        df["rsi14"] = ta.momentum.rsi(df["Close"], window=14)
        macd = ta.trend.macd(df["Close"])
        macd_signal = ta.trend.macd_signal(df["Close"])
        macd_diff = ta.trend.macd_diff(df["Close"])
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_diff"] = macd_diff
        df["atr14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
        return df

    df = add_technical_features(df)

    print('[DEBUG] After technical features shape:', df.shape)
    print('[DEBUG] RAM used:', psutil.virtual_memory().percent, '%')

    # --- 2. Feature interaction (pairwise product/ratio of top features) ---
    def add_feature_interactions(df, topn=5):
        num_cols = [c for c in df.columns if df[c].dtype != 'O' and c not in ['target','pred_proba']]
        top = num_cols[:topn]
        for i in range(len(top)):
            for j in range(i+1, len(top)):
                f1, f2 = top[i], top[j]
                df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
                df[f'{f1}_div_{f2}'] = df[f1] / (df[f2]+1e-6)
        return df

    df = add_feature_interactions(df)

    print('[DEBUG] After feature interactions shape:', df.shape)
    print('[DEBUG] RAM used:', psutil.virtual_memory().percent, '%')

    # --- 3. Auto feature generation (Featuretools) ---
    if ft is not None:
        df = df.copy()
        # Remove any duplicate index if exists
        if 'auto_id' in df.columns:
            df = df.drop(columns=['auto_id'])
        df['auto_id'] = np.arange(len(df))  # Ensure unique index for Featuretools
        es = ft.EntitySet(id='data')
        es = es.add_dataframe(dataframe_name='main', dataframe=df.reset_index(drop=True), index='auto_id')
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='main', max_depth=2, verbose=True)
        df_ft = feature_matrix
        print(f"[INFO] Featuretools generated: {df_ft.shape[1]} features")
        df_ft.to_parquet(OUT_PATH)
    else:
        print("[WARN] Featuretools not installed, skipping auto feature generation.")
        df_ft = df

    print('[DEBUG] After Featuretools shape:', df_ft.shape)
    print('[DEBUG] RAM used:', psutil.virtual_memory().percent, '%')

    # --- 4. Auto feature extraction (tsfresh) ---
    if extract_features is not None and 'Date' in df.columns:
        df_tsf = df.copy()
        # --- Ensure all columns are numeric for tsfresh ---
        df_tsf = df_tsf.apply(pd.to_numeric, errors='coerce')
        df_tsf = df_tsf.replace([np.inf, -np.inf], np.nan).fillna(0)
        # ปรับ id ให้แบ่งเป็นหลายกลุ่มเพื่อ parallel จริง
        batch_size = 30000
        df_tsf['id'] = df_tsf.index // batch_size
        df_tsf['time'] = pd.to_datetime(df_tsf['Date']).astype(int) // 10**9
        # --- เพิ่ม logic debug mode ---
        if debug_mode:
            print('[DEBUG] Sampling tsfresh input for debug mode...')
            df_tsf = df_tsf.sample(n=min(5000, len(df_tsf)), random_state=42)
            extraction_settings = MinimalFCParameters() if MinimalFCParameters is not None else None
        else:
            extraction_settings = None
        import multiprocessing
        n_jobs = max(1, min(8, multiprocessing.cpu_count() - 1))
        print(f'[DEBUG] tsfresh input shape: {df_tsf.shape}, columns: {list(df_tsf.columns)}')
        print(f'[DEBUG] Start tsfresh feature extraction with n_jobs={n_jobs}, batch_size={batch_size}...')
        try:
            extracted = extract_features(
                df_tsf,
                column_id='id',
                column_sort='time',
                default_fc_parameters=extraction_settings,
                disable_progressbar=False,
                n_jobs=n_jobs
            )
            print('[DEBUG] tsfresh feature extraction done.')
            print(f'[DEBUG] tsfresh output shape: {extracted.shape}, columns: {list(extracted.columns)})')
            extracted.to_parquet(os.path.join(OUT_DIR, 'tsfresh_features.parquet'))
        except MemoryError as e:
            print('[ERROR] tsfresh feature extraction failed: Out of memory!')
            import traceback; traceback.print_exc()
        except KeyboardInterrupt:
            print('[ERROR] tsfresh feature extraction interrupted by user.')
        except Exception as e:
            print('[ERROR] tsfresh feature extraction failed:', e)
            import traceback; traceback.print_exc()
        print('[DEBUG] RAM used after tsfresh:', psutil.virtual_memory().percent, '%')
    else:
        print("[WARN] tsfresh not installed or no Date column, skipping tsfresh features.")

    # --- 5. Summary/plot ---
    summary = pd.DataFrame({'feature': df_ft.columns, 'nunique': [df_ft[c].nunique() for c in df_ft.columns]})
    summary.to_csv(os.path.join(OUT_DIR, 'auto_features_summary.csv'), index=False)
    try:
        summary['nunique'].plot(kind='hist', bins=30, title='Feature Uniqueness Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'auto_features_uniqueness.png'))
        plt.close()
    except Exception as e:
        print(f"[WARN] Plotting uniqueness histogram failed: {e}")

    print("[เทพ] Auto feature generation เสร็จสมบูรณ์! ดูไฟล์ output_default/auto_features.parquet, tsfresh_features.parquet, auto_features_summary.csv, auto_features_uniqueness.png")

if __name__ == "__main__":
    import sys
    debug_mode = False
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        debug_mode = True
    main(debug_mode=debug_mode)
