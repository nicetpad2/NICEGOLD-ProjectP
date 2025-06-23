import locale
from dateutil.parser import parse as parse_date
import pandas as pd
from typing import Union, Optional, List, Any

THAI_MONTH_MAP = {
    "ม.ค.": "01",
    "ก.พ.": "02",
    "มี.ค.": "03",
    "เม.ย.": "04",
    "พ.ค.": "05",
    "มิ.ย.": "06",
    "ก.ค.": "07",
    "ส.ค.": "08",
    "ก.ย.": "09",
    "ต.ค.": "10",
    "พ.ย.": "11",
    "ธ.ค.": "12",
}

def robust_date_parser(date_string):
    normalized = str(date_string)
    for th, num in THAI_MONTH_MAP.items():
        if th in normalized:
            normalized = normalized.replace(th, num)
            break
    try:
        dt = parse_date(normalized, dayfirst=True)
    except Exception as e:
        raise ValueError(f"Cannot parse Thai date: {date_string}") from e
    if dt.year > 2500:
        dt = dt.replace(year=dt.year - 543)
    return dt

def normalize_thai_date(ts: str) -> str:
    # ...existing logic from data_loader.py...
    return ts

def parse_datetime_safely(value: pd.Series, default: Optional[Any] = None) -> pd.Series:
    """
    Parse datetime from pandas Series safely with multiple format support.
    
    Args:
        value: A pandas Series containing datetime strings
        default: Value to use if parsing fails
        
    Returns:
        pandas Series with parsed datetime values
        
    Raises:
        TypeError: If input is not a pandas Series
    """
    if not isinstance(value, pd.Series):
        raise TypeError("Expected pandas.Series input")
        
    # Try to parse with pandas to_datetime
    try:
        return pd.to_datetime(value, errors='coerce')
    except:
        pass
    
    # If that fails, try to parse each value individually with format inference
    result = pd.Series(index=value.index, dtype='datetime64[ns]')
    
    for idx, val in value.items():
        try:
            if pd.isna(val):
                result.loc[idx] = pd.NaT
                continue
                
            # Handle Thai Buddhist calendar dates
            str_val = str(val)
            if len(str_val) >= 4 and str_val[0:2] == '25':  # Likely Thai year starting with 25xx
                year = int(str_val[0:4])
                if year >= 2500:
                    # Convert to Western year
                    western_year = year - 543
                    # Replace the year part
                    str_val = f"{western_year}{str_val[4:]}"

            parsed = pd.to_datetime(str_val, format=None)
            result.loc[idx] = parsed
        except:
            result.loc[idx] = default if default is not None else pd.NaT
    
    return result

def prepare_datetime(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Prepare datetime index and convert Thai Buddhist dates to Western dates.
    
    Args:
        df: DataFrame with 'Date' and 'Timestamp' columns
        timeframe: Timeframe string (e.g., 'M1', 'M15')
        
    Returns:
        DataFrame with datetime index
    """
    result = df.copy()
    
    # Check for required columns
    datetime_col = None
    separate_columns = False
    
    # Case 1: Separate Date and Time columns
    if 'Date' in result.columns and 'Time' in result.columns:
        separate_columns = True
        # Create a DateTime column
        result['DateTime'] = result['Date'] + ' ' + result['Time']
        datetime_col = 'DateTime'
    
    # Case 2: Single DateTime or Timestamp column
    elif 'DateTime' in result.columns:
        datetime_col = 'DateTime'
    elif 'Timestamp' in result.columns:
        datetime_col = 'Timestamp'
    elif '\ufeffTimestamp' in result.columns:  # Handle BOM in column name
        datetime_col = '\ufeffTimestamp'
    
    if datetime_col is None:
        # Cannot process without datetime information
        return result
    
    # Convert datetime column to datetime type
    result[datetime_col] = pd.to_datetime(result[datetime_col], errors='coerce')
    
    # Check for Thai Buddhist calendar dates (BE) and convert to CE
    mask = result[datetime_col].dt.year > 2500
    if mask.any():
        result.loc[mask, datetime_col] = result.loc[mask, datetime_col].apply(
            lambda x: x.replace(year=x.year - 543) if pd.notnull(x) else x
        )
    
    # Set the datetime column as index
    result = result.set_index(datetime_col)
    
    # Sort index
    result = result.sort_index()
    
    return result# Convert Date column if it looks like Thai Buddhist calendar (year >= 2500)
    if 'Date' in result.columns:
        dates = []
        for date_str in result['Date']:
            try:
                # Parse as string first
                date_str = str(date_str)
                
                # Check if it's likely a Thai Buddhist date
                if len(date_str) >= 4:
                    year_part = date_str[:4] if date_str[0].isdigit() else date_str[:2]
                    if year_part.isdigit() and int(year_part) >= 2500:
                        # It's a Thai Buddhist date, convert year
                        year = int(date_str[:4])
                        western_year = year - 543
                        date_str = f"{western_year}{date_str[4:]}"
                
                dates.append(date_str)
            except:
                dates.append(date_str)  # Keep original if parsing fails
                
        result['Date'] = dates
    
    # Combine Date and Timestamp if both exist
    if 'Date' in result.columns and 'Timestamp' in result.columns:
        # Convert to datetime
        try:
            result['datetime'] = pd.to_datetime(
                result['Date'].astype(str) + ' ' + result['Timestamp'].astype(str),
                errors='coerce'
            )
            result = result.set_index('datetime')
        except Exception as e:
            # Fallback: try to parse each component separately
            try:
                date_col = pd.to_datetime(result['Date'], errors='coerce')
                time_col = result['Timestamp'].astype(str)
                
                # Combine date and time
                result['datetime'] = pd.to_datetime(
                    date_col.dt.strftime('%Y-%m-%d') + ' ' + time_col,
                    errors='coerce'
                )
                result = result.set_index('datetime')
            except:
                # If all else fails, create a dummy index
                pass
    
    return result

def convert_thai_years(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Thai Buddhist years (BE) to Western years (CE)"""
    result = df.copy()
    
    # Look for date columns
    date_cols = []
    for col in result.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
            date_cols.append(col)
    
    # Process each date column
    for col in date_cols:
        try:
            # Try to convert to datetime
            result[col] = pd.to_datetime(result[col], errors='coerce')
            
            # Check for Thai years (year > 2500)
            mask = result[col].dt.year > 2500
            if mask.any():
                # Convert Thai years to Western
                result.loc[mask, col] = result.loc[mask, col].apply(
                    lambda x: x.replace(year=x.year - 543) if pd.notnull(x) else x
                )
        except:
            pass
            
    return result

def convert_thai_datetime(dt_value):
    """Convert a single Thai datetime value to Western calendar"""
    if pd.isna(dt_value):
        return dt_value
        
    try:
        if isinstance(dt_value, str):
            dt_value = pd.to_datetime(dt_value, errors='coerce')
            
        if hasattr(dt_value, 'year') and dt_value.year > 2500:
            return dt_value.replace(year=dt_value.year - 543)
        return dt_value
    except:
        return dt_value

def parse_datetime(value, formats=None):
    """
    Parse datetime with multiple format support
    
    Args:
        value: String or value to parse
        formats: List of datetime formats to try
        
    Returns:
        Parsed datetime or None if parsing fails
    """
    if pd.isna(value):
        return None
        
    str_val = str(value)
    
    # If formats provided, try them one by one
    if formats:
        for fmt in formats:
            try:
                return pd.to_datetime(str_val, format=fmt)
            except:
                continue
    
    # Try default pandas parsing
    try:
        return pd.to_datetime(str_val, format=None)
    except:
        return None
