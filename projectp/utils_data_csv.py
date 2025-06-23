from utils import prepare_csv_auto
from src.utils.data_utils import convert_thai_datetime
import pandas as pd

def load_and_prepare_main_csv(path, add_target=False, target_func=None, rename_timestamp=True):
    """
    Load CSV with Thai Buddhist Timestamp, convert to datetime, add target column if needed, and rename column for pipeline compatibility.
    Args:
        path: str, path to CSV
        add_target: bool, whether to add a 'target' column
        target_func: callable, function to generate target (df) -> Series
        rename_timestamp: bool, rename 'Timestamp' to 'timestamp'
    Returns:
        df: pd.DataFrame (ready for pipeline)
    """
    df = prepare_csv_auto(path)
    # แปลง Timestamp พ.ศ. เป็น datetime ค.ศ. (inplace)
    df = convert_thai_datetime(df, "Timestamp")
    if rename_timestamp and "Timestamp" in df.columns:
        df = df.rename(columns={"Timestamp": "timestamp"})
    # เติม target อัตโนมัติถ้าต้องการ
    if add_target and "target" not in df.columns:
        if target_func:
            df["target"] = target_func(df)
        else:
            # Default: next close > close = 1 else 0
            df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df
