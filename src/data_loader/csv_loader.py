

# Avoid circular imports by using delayed imports
        from projectp.utils_feature import (
from src.data_loader.date_utils import normalize_thai_date
from typing import Any, Dict, Optional
import os
import pandas as pd
def _get_column_mapping_functions():
    """Get column mapping functions with delayed import to avoid circular imports"""
    try:
            assert_no_lowercase_columns, 
            map_standard_columns, 
        )

        return map_standard_columns, assert_no_lowercase_columns
    except ImportError:
        # Fallback functions to avoid circular import
        def map_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
            """Fallback function when utils_feature is not available"""
            return df

        def assert_no_lowercase_columns(df: pd.DataFrame) -> None:
            """Fallback function when utils_feature is not available"""
            pass

        return map_standard_columns, assert_no_lowercase_columns


def safe_load_csv_auto(
    file_path: str, row_limit: Optional[int] = None, **kwargs: Any
) -> pd.DataFrame:
    """
    Load CSV with automatic processing including datetime detection and deduplication
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {file_path}")

    # Read CSV file
    df = pd.read_csv(file_path, nrows = row_limit, **kwargs)

    # Handle BOM in column names
    if df.columns[0].startswith("\ufeff"):
        df.columns = [col.replace("\ufeff", "") for col in df.columns]

    # Map standard columns using delayed import
    try:
        map_standard_columns, assert_no_lowercase_columns = (
            _get_column_mapping_functions()
        )
        df = map_standard_columns(df)
        assert_no_lowercase_columns(df)
    except Exception:
        pass  # If mapping functions not available, continue without them

    # Auto - detect and merge datetime columns
    datetime_columns_found = []
    main_datetime_col = None
    # Check for separate Date and Time columns
    if "Date" in df.columns and "Time" in df.columns:
        # Merge Date and Time
        df["datetime"] = df["Date"].astype(str) + " " + df["Time"].astype(str)
        datetime_columns_found.append("datetime")
        main_datetime_col = "datetime"
    elif "Date" in df.columns and "Timestamp" in df.columns:
        # Merge Date and Timestamp
        df["datetime"] = df["Date"].astype(str) + " " + df["Timestamp"].astype(str)
        datetime_columns_found.append("datetime")
        main_datetime_col = "datetime"

    # Check for Timestamp column
    if "Timestamp" in df.columns and main_datetime_col is None:
        datetime_columns_found.append("Timestamp")
        main_datetime_col = "Timestamp"

    # Check for other datetime - like columns
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            if col not in datetime_columns_found:
                # Try to parse as datetime
                try:
                    test_sample = df[col].dropna().head(5)
                    if not test_sample.empty:
                        pd.to_datetime(test_sample.iloc[0])
                        datetime_columns_found.append(col)
                        if main_datetime_col is None:
                            main_datetime_col = col
                except:
                    pass

    # Convert datetime columns and handle Thai Buddhist years
    for col in datetime_columns_found:
        if col in df.columns:
            # Handle Thai year conversion (Buddhist to Gregorian)
            try:
                df[col] = df[col].astype(str).apply(normalize_thai_date)
            except:
                pass

            # Convert to datetime
            try:
                df[col] = pd.to_datetime(df[col], errors = "coerce")
            except:
                pass

    # Remove duplicates if there are any datetime columns
    if main_datetime_col and len(df) > 1:
        # Sort by main datetime column and remove duplicates
        if main_datetime_col in df.columns:
            try:
                df = df.sort_values(main_datetime_col)
                initial_len = len(df)
                df = df.drop_duplicates(subset = [main_datetime_col], keep = "last")
                final_len = len(df)
                if initial_len > final_len:
                    print(f"Removed {initial_len - final_len} duplicate rows")
            except:
                pass

    # Set datetime index if main datetime column is available
    if main_datetime_col and main_datetime_col in df.columns:
        try:
            df = df.set_index(main_datetime_col)
            if df.index.dtype != "datetime64[ns]":
                df.index = pd.to_datetime(df.index)
        except:
            pass

    return df


def safe_load_csv(path: str, fill_method: str = "ffill") -> pd.DataFrame:
    """
    Load CSV file with deduplication and datetime processing
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {path}")

    # Use safe_load_csv_auto for better processing
    df = safe_load_csv_auto(path)

    # Apply fill method with new pandas syntax
    if fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()
    elif fill_method:
        # For backward compatibility, try old method but catch deprecation warning
        try:
            df = df.fillna(method = fill_method)
        except:
            df = df.ffill()  # Default fallback

    return df


def read_csv_with_date_parse(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates = True)

    # Use delayed import to avoid circular imports
    try:
        map_standard_columns, assert_no_lowercase_columns = (
            _get_column_mapping_functions()
        )
        df = map_standard_columns(df)
        assert_no_lowercase_columns(df)
    except Exception:
        pass  # Continue without column mapping if functions not available

    return df


def load_data_from_csv(file_path: str, nrows: int = None, auto_convert: bool = True):
    temp_df = pd.read_csv(file_path, nrows = nrows)

    # Use delayed import to avoid circular imports
    try:
        map_standard_columns, assert_no_lowercase_columns = (
            _get_column_mapping_functions()
        )
        temp_df = map_standard_columns(temp_df)
        assert_no_lowercase_columns(temp_df)
    except Exception:
        pass  # Continue without column mapping if functions not available

    # ...existing code...
    return temp_df


def read_csv_in_chunks(path: str, chunksize: int = 100_000, **kwargs):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {path}")
    for chunk in pd.read_csv(path, chunksize = chunksize, **kwargs):
        # Use delayed import to avoid circular imports
        try:
            map_standard_columns, assert_no_lowercase_columns = (
                _get_column_mapping_functions()
            )
            chunk = map_standard_columns(chunk)
            assert_no_lowercase_columns(chunk)
        except Exception:
            pass  # Continue without column mapping if functions not available
        yield chunk


def validate_csv_data(df, required_cols):
    """Validate that required columns exist in the DataFrame."""
    if required_cols is None:
        return True
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True