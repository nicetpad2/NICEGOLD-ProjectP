
from typing import Any, Optional
    import datetime
    import json
import logging
import numpy as np
    import os
import pandas as pd
    import sys
def safe_set_datetime(df: pd.DataFrame, idx: Any, col: str, val: Any, naive_tz: Optional[str] = None) -> None:
    """
    Safely set a datetime value in a DataFrame at a given index and column, ensuring dtype and handling errors robustly.
    Ensures dtype is exactly 'datetime64[ns]' (no timezone).
    Args:
        df (pd.DataFrame): The DataFrame to modify.
        idx (Any): The index/row to set.
        col (str): The column name.
        val (Any): The value to set (should be convertible to datetime).
        naive_tz (str, optional): If provided, localize naive datetimes to this timezone then convert to UTC.
    """
    try:
        # Ensure column exists
        if col not in df.columns:
            df[col] = pd.NaT

        # Convert value to datetime
        try:
            dt_val = pd.to_datetime(val, errors = 'coerce')

            # Handle timezone conversion
            if naive_tz and hasattr(dt_val, 'tz') and dt_val.tz is None:
                # Localize naive datetime to specified timezone, then convert to UTC
                dt_val = dt_val.tz_localize(naive_tz).tz_convert('UTC')

            # Remove timezone info if present to make it naive
            if hasattr(dt_val, 'tz_localize') and getattr(dt_val, 'tzinfo', None) is not None:
                dt_val = dt_val.tz_localize(None)

        except Exception as e:
            logging.warning(f"safe_set_datetime: Failed to convert value '{val}' to datetime: {e}")
            dt_val = pd.NaT

        # Ensure column dtype is datetime64[ns] (no timezone)
        if not (df[col].dtype == 'datetime64[ns]'):
            try:
                df[col] = pd.to_datetime(df[col], errors = 'coerce')
            except Exception:
                pass

        # Cast value to match column dtype before assignment
        try:
            dt_val = pd.to_datetime(dt_val)
        except Exception:
            dt_val = pd.NaT

        try:
            df.at[idx, col] = dt_val
        except Exception as e:
            # If index is missing, try fallback to iloc if possible
            logging.warning(f"safe_set_datetime: Failed to set value at idx = {idx}, col = {col}: {e}")
            try:
                if isinstance(idx, int) and 0 <= idx < len(df):
                    df.iloc[idx, df.columns.get_loc(col)] = dt_val
                else:
                    # If index is out of bounds, do nothing
                    pass
            except Exception as e2:
                logging.error(f"safe_set_datetime: Fallback set failed: {e2}")

        # After setting, force dtype to datetime64[ns] if possible
        try:
            if not (df[col].dtype == 'datetime64[ns]'):
                df[col] = pd.to_datetime(df[col], errors = 'coerce')
        except Exception:
            pass

    except Exception as e:
        logging.error(f"safe_set_datetime: Unexpected error: {e}")

def simple_converter(o):  # pragma: no cover
    """Converts common pandas/numpy types for JSON serialization."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        if np.isnan(o):
            return None
        if np.isinf(o):
            return "Infinity" if o > 0 else " - Infinity"
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, pd.Timedelta):
        return str(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if pd.isna(o):
        return None
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)

def safe_get_global(var_name, default_value):
    """
    Safely retrieves a global variable from the current module's scope.
    Args:
        var_name (str): The name of the global variable to retrieve.
        default_value: The value to return if the global variable is not found.
    Returns:
        The value of the global variable or the default value.
    """
    try:
        return globals().get(var_name, default_value)
    except Exception as e:
        logging.error(f"   (Error) Unexpected error in safe_get_global for '{var_name}': {e}", exc_info = True)
        return default_value

def setup_output_directory(base_dir, dir_name):
    """
    Creates the output directory if it doesn't exist and checks write permissions.
    Args:
        base_dir (str): The base directory path.
        dir_name (str): The name of the output directory to create within the base directory.
    Returns:
        str: The full path to the created/verified output directory.
    Raises:
        SystemExit: If the directory cannot be created or written to.
    """
    output_path = os.path.join(base_dir, dir_name)
    logging.info(f"   (Setup) กำลังตรวจสอบ/สร้าง Output Directory: {output_path}")
    try:
        os.makedirs(output_path, exist_ok = True)
        logging.info(f"      -> Directory exists or was created.")
        # Test write permissions
        test_file_path = os.path.join(output_path, ".write_test")
        with open(test_file_path, "w", encoding = "utf - 8") as f:
            f.write("test")
        try:
            os.remove(test_file_path)
        except OSError:
            logging.debug(f"Unable to remove test file {test_file_path}")
        logging.info(f"      -> การเขียนไฟล์ทดสอบสำเร็จ.")
        return output_path
    except OSError as e:
        logging.error(f"   (Error) ไม่สามารถสร้างหรือเขียนใน Output Directory '{output_path}': {e}", exc_info = True)
        sys.exit(f"   ออก: ปัญหาการเข้าถึง Output Directory ({output_path}).")

def load_app_config(*args, **kwargs):
    raise NotImplementedError('load_app_config is not implemented')