"""
Utility functions for gold data conversion
"""
import os
import pandas as pd
import logging
import glob
from typing import Union, Optional, List

logger = logging.getLogger(__name__)

def auto_convert_csv_to_parquet(file_path: str, output_path: Optional[str] = None, **kwargs) -> Optional[str]:
    """
    Convert CSV to Parquet format
    
    Args:
        file_path (str): Path to the CSV file
        output_path (str, optional): Output path for Parquet file
        **kwargs: Additional parameters for CSV reading
        
    Returns:
        str: Path to the created parquet file or None if failed
    """
    try:
        # Default output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            dir_name = os.path.dirname(file_path)
            output_path = os.path.join(dir_name, f"{base_name}.parquet")
        
        # Read CSV file
        df = pd.read_csv(file_path, **kwargs)
        
        # Convert to parquet
        df.to_parquet(output_path, index=False)
        logger.info(f"Converted {file_path} to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error in auto_convert_csv_to_parquet: {str(e)}")
        return None

def auto_convert_gold_csv(input_path: str, output_path: Optional[str] = None) -> None:
    """
    Convert western date to Thai date for gold CSV files
    
    Args:
        input_path: Directory or file path containing gold CSV files
        output_path: Output directory or file path. If None, appends '_thai' to input filename
    """
    try:
        # Handle input path
        if not input_path:
            input_path = "."  # Current directory
            
        if os.path.isdir(input_path):
            # Find all CSV files matching pattern
            pattern = os.path.join(input_path, "XAUUSD_M*.csv")
            files = glob.glob(pattern)
            
            # Handle each file
            for file_path in files:
                file_name = os.path.basename(file_path)
                if "_thai" not in file_name:  # Skip already converted files
                    if os.path.isdir(output_path):
                        # If output is directory, create file in that directory
                        out_file = os.path.join(output_path, file_name.replace(".csv", "_thai.csv"))
                    else:
                        # Default output file
                        out_file = os.path.join(os.path.dirname(file_path), file_name.replace(".csv", "_thai.csv"))
                    
                    _convert_single_file(file_path, out_file)
        else:
            # Single file processing
            if not output_path:
                output_path = input_path.replace(".csv", "_thai.csv")
                
            _convert_single_file(input_path, output_path)
    
    except Exception as e:
        logger.error(f"Error in auto_convert_gold_csv: {str(e)}")


def _convert_single_file(input_file: str, output_file: str) -> None:
    """Helper function to convert a single gold CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        if df.empty:
            # Don't create any output file for empty input
            return
        
        # Check for required columns
        if not _has_required_time_columns(df):
            logger.warning(f"File {input_file} missing required time columns")
            # Don't create any output file when required columns are missing
            return
        
        # Check for invalid date in 'Date' column if it exists
        if 'Date' in df.columns:
            try:
                # Try to parse dates
                pd.to_datetime(df['Date'], errors='raise')
            except:
                logger.warning(f"File {input_file} contains invalid dates")
                # Don't create any output file for invalid dates
                return
                return
            
        # Process the dataframe
        df = _process_gold_dataframe(df)
        
        # Save to output file
        df.to_csv(output_file, index=False)
        logger.info(f"Converted {input_file} to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")
        # Create empty output file with proper columns in case of error
        pd.DataFrame(columns=['Date', 'Timestamp', 'Open', 'High', 'Low', 'Close']).to_csv(output_file, index=False)


def _has_required_time_columns(df: pd.DataFrame) -> bool:
    """Check if dataframe has required time columns"""
    # Check for common date/time column combinations
    time_columns = [
        ["Date", "Time"],
        ["Timestamp"],
        ["DateTime"],
        ["\ufeffTimestamp"],  # With BOM
        ["Date/Time (UTC)"]
    ]
    
    for col_set in time_columns:
        if all(col in df.columns for col in col_set):
            return True
            
    required_cols = ["Open", "High", "Low", "Close"]
    if not all(any(col.lower() == req.lower() for col in df.columns) for req in required_cols):
        return False
        
    return False


def _process_gold_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process gold dataframe to convert dates and standardize column names"""
    # Make a copy to avoid modifying input
    result = df.copy()
    
    # Standardize column names
    result = _standardize_columns(result)
    
    # Process date/time information
    result = _process_datetime(result)
    
    return result


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names"""
    # Copy the dataframe
    result = df.copy()
      # Map for column standardization
    col_map = {
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'Time': 'Timestamp',  # Rename Time to Timestamp for standardization
        '\ufeffTimestamp': 'Timestamp'  # BOM character
    }
    
    # Rename columns
    for old, new in col_map.items():
        if old in result.columns:
            result = result.rename(columns={old: new})
    
    return result


def _process_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Process datetime columns"""
    result = df.copy()

    # Handle different datetime column formats
    if "DateTime" in result.columns:
        # Split DateTime into Date and Timestamp
        result[["Date", "Timestamp"]] = result["DateTime"].str.split(" ", n=1, expand=True)
        result = result.drop("DateTime", axis=1)

    elif "Date/Time (UTC)" in result.columns:
        # Split DateTime into Date and Timestamp
        result[["Date", "Timestamp"]] = result["Date/Time (UTC)"].str.split(" ", n=1, expand=True)
        result = result.drop("Date/Time (UTC)", axis=1)

    elif "Timestamp" in result.columns and " " in str(result["Timestamp"].iloc[0]):
        # Split Timestamp into Date and Timestamp if it contains both
        result[["Date", "Timestamp"]] = result["Timestamp"].str.split(" ", n=1, expand=True)

    # Process Date column if it exists
    if "Date" in result.columns:
        # Convert western year to Thai year (add 543)
        try:
            # Parse various date formats
            date_col = result["Date"].copy()

            # Convert to pandas datetime
            pd_date = pd.to_datetime(date_col, errors='coerce')

            # Check for valid dates
            valid_mask = ~pd_date.isna()

            # If no valid dates at all, return empty dataframe with headers
            if not valid_mask.any():
                return pd.DataFrame(columns=result.columns)

            # Get year and add 543 for Thai year
            years = pd_date.dt.year + 543

            # Format the new date
            result.loc[valid_mask, "Date"] = (
                years.astype(str) + "-" + 
                pd_date.dt.month.astype(str).str.zfill(2) + "-" + 
                pd_date.dt.day.astype(str).str.zfill(2)
            )

            # Remove rows with invalid dates
            result = result[valid_mask].reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error converting dates: {str(e)}")
            # Return empty DataFrame with headers on error
            return pd.DataFrame(columns=result.columns)

    return result


def auto_convert_csv_to_parquet(csv_path: str, output_dir: str) -> None:
    """
    Convert CSV file to Parquet format
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save the Parquet file
    """
    try:
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the base filename without extension
        base_name = os.path.basename(csv_path).replace('.csv', '')
        
        # Define the output path
        parquet_path = os.path.join(output_dir, f"{base_name}.parquet")
        
        # Check if Parquet libraries are available
        try:
            # Try to import pyarrow or fastparquet
            try:
                import pyarrow
                engine = "pyarrow"
            except ImportError:
                import fastparquet
                engine = "fastparquet"
                
            # Read the CSV
            df = pd.read_csv(csv_path)
            
            # Write to Parquet
            df.to_parquet(parquet_path, engine=engine)
            
            logger.info(f"Converted {csv_path} to {parquet_path}")
            
        except ImportError:
            logger.warning("Neither pyarrow nor fastparquet available. Cannot convert to Parquet.")
            
    except Exception as e:
        logger.error(f"Error converting CSV to Parquet: {str(e)}")
