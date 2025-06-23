import os
import hashlib
import pandas as pd

def load_main_training_data(config=None, default_path='output_default/preprocessed.csv'):
    """
    Load, validate, and log main training data file (production-ready, multi-format, versioning-ready)
    Args:
        config: dict, may contain 'train_data_path'
        default_path: fallback path
    Returns:
        df: pd.DataFrame
        info: dict (path, shape, columns, hash, summary)
    """
    path = config.get('train_data_path') if config and 'train_data_path' in config else default_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data file not found: {path}")
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext == '.parquet':
        df = pd.read_parquet(path)
    elif ext == '.feather':
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported training data file format: {ext}")
    # Validate schema/columns
    required_cols = ['target']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in training data!")
    # Log/Export summary
    file_hash = hashlib.md5(open(path, 'rb').read()).hexdigest()
    info = {
        'path': path,
        'shape': df.shape,
        'columns': list(df.columns),
        'hash': file_hash,
        'head': df.head(2).to_dict(),
    }
    # จุด hook: data versioning, data catalog, lineage, logging
    # ตัวอย่าง: export log
    with open('output_default/train_data_info.json', 'w') as f:
        import json; json.dump(info, f, indent=2)
    return df, info
