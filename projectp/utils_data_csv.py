
from projectp.data_validator import RealDataValidator
from src.utils.data_utils import convert_thai_datetime
from utils import prepare_csv_auto
import os
import pandas as pd
def load_and_prepare_main_csv(path, add_target = False, target_func = None, rename_timestamp = True):
    """
    Load CSV with Thai Buddhist Timestamp, convert to datetime, add target column if needed, and rename column for pipeline compatibility.
    ENFORCES REAL DATA ONLY - validates data comes from datacsv folder.

    Args:
        path: str, path to CSV (must be in datacsv folder)
        add_target: bool, whether to add a 'target' column
        target_func: callable, function to generate target (df) -> Series
        rename_timestamp: bool, rename 'Timestamp' to 'timestamp'
    Returns:
        df: pd.DataFrame (ready for pipeline)
    """
    # Validate that path is from datacsv folder or validate the data
    validator = RealDataValidator()

    # Check if this is a path to datacsv folder file
    datacsv_folder = validator.datacsv_path
    if not path.startswith(datacsv_folder):
        # If not a direct datacsv path, validate it's real data
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ CRITICAL: Data file not found: {path}")

        # Load and validate it's real data
        df_test = pd.read_csv(path)
        if len(df_test) < 100:  # Basic check for real data
            raise ValueError(f"❌ CRITICAL: File {path} appears to be dummy data (too few rows)")

    df = prepare_csv_auto(path)
    # แปลง Timestamp พ.ศ. เป็น datetime ค.ศ. (inplace)
    df = convert_thai_datetime(df, "Timestamp")
    if rename_timestamp and "Timestamp" in df.columns:
        df = df.rename(columns = {"Timestamp": "timestamp"})
    # เติม target อัตโนมัติถ้าต้องการ
    if add_target and "target" not in df.columns:
        if target_func:
            df["target"] = target_func(df)
        else:
            # Default: next close > close = 1 else 0
            df["target"] = (df["Close"].shift( - 1) > df["Close"]).astype(int)
    return df