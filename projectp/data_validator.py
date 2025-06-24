"""
Data Validation Module for ProjectP
Ensures only real data from datacsv folder is used in all pipeline modes.
No dummy, synthetic, or test data allowed.
"""

import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
from projectp.pro_log import pro_log


class RealDataValidator:
    """
    Validator to ensure only real data from datacsv folder is used.
    Prevents any fallback to dummy or synthetic data.
    """
    
    def __init__(self, datacsv_path: str = None):
        """Initialize validator with datacsv folder path"""
        if datacsv_path is None:
            # Default to projectp/datacsv
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.datacsv_path = os.path.join(base_dir, "datacsv")
        else:
            self.datacsv_path = datacsv_path
            
        self.required_columns = ["Open", "High", "Low", "Close", "Volume"]
        
    def validate_datacsv_folder(self) -> bool:
        """
        Validate that datacsv folder exists and contains real data files.
        Returns True if valid, raises exception if invalid.
        """
        pro_log("🔍 Validating datacsv folder for real data...", tag="DataValidator")
        
        # Check if datacsv folder exists
        if not os.path.exists(self.datacsv_path):
            error_msg = f"❌ CRITICAL: datacsv folder not found at {self.datacsv_path}. Pipeline cannot proceed without real data."
            pro_log(error_msg, level="error", tag="DataValidator")
            raise FileNotFoundError(error_msg)
            
        if not os.path.isdir(self.datacsv_path):
            error_msg = f"❌ CRITICAL: {self.datacsv_path} is not a directory. Pipeline requires datacsv folder with real data."
            pro_log(error_msg, level="error", tag="DataValidator")
            raise NotADirectoryError(error_msg)
            
        # List CSV files in datacsv
        csv_files = [f for f in os.listdir(self.datacsv_path) if f.endswith('.csv')]
        
        if not csv_files:
            error_msg = f"❌ CRITICAL: No CSV files found in {self.datacsv_path}. Pipeline requires real data files."
            pro_log(error_msg, level="error", tag="DataValidator")
            raise ValueError(error_msg)
            
        pro_log(f"✅ Found {len(csv_files)} CSV files in datacsv: {csv_files}", tag="DataValidator")
        
        # Validate each CSV file contains real data
        valid_files = []
        for csv_file in csv_files:
            file_path = os.path.join(self.datacsv_path, csv_file)
            if self._validate_csv_file(file_path):
                valid_files.append(csv_file)
                
        if not valid_files:
            error_msg = f"❌ CRITICAL: No valid real data files found in {self.datacsv_path}."
            pro_log(error_msg, level="error", tag="DataValidator")
            raise ValueError(error_msg)
            
        pro_log(f"✅ Validated {len(valid_files)} real data files: {valid_files}", tag="DataValidator")
        return True
        
    def _validate_csv_file(self, file_path: str) -> bool:
        """Validate that a CSV file contains real trading data"""
        try:
            # Check file size (real data should be substantial)
            file_size = os.path.getsize(file_path)
            if file_size < 1024:  # Less than 1KB is suspicious
                pro_log(f"⚠️ WARNING: {os.path.basename(file_path)} is very small ({file_size} bytes)", tag="DataValidator")
                return False
                
            # Load and check data structure
            df = pd.read_csv(file_path)
            
            # Check minimum rows (real data should have substantial history)
            if len(df) < 100:
                pro_log(f"⚠️ WARNING: {os.path.basename(file_path)} has only {len(df)} rows", tag="DataValidator")
                return False
                
            # Check for required columns (case insensitive)
            df_cols_lower = [col.lower() for col in df.columns]
            required_found = 0
            for req_col in ["open", "high", "low", "close"]:  # Volume is optional
                if req_col in df_cols_lower:
                    required_found += 1
                    
            if required_found < 4:
                pro_log(f"⚠️ WARNING: {os.path.basename(file_path)} missing required OHLC columns", tag="DataValidator")
                return False
                
            # Check for realistic price ranges (not dummy data like all 1.0)
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if df[col].nunique() == 1:  # All values the same
                    pro_log(f"⚠️ WARNING: {os.path.basename(file_path)} column {col} has constant values (dummy data?)", tag="DataValidator")
                    return False
                    
            pro_log(f"✅ {os.path.basename(file_path)} validated as real data", tag="DataValidator")
            return True
            
        except Exception as e:
            pro_log(f"❌ ERROR validating {os.path.basename(file_path)}: {e}", level="error", tag="DataValidator")
            return False
            
    def get_available_data_files(self) -> List[str]:
        """Get list of available real data files in datacsv folder"""
        self.validate_datacsv_folder()
        csv_files = [f for f in os.listdir(self.datacsv_path) if f.endswith('.csv')]
        valid_files = []
        
        for csv_file in csv_files:
            file_path = os.path.join(self.datacsv_path, csv_file)
            if self._validate_csv_file(file_path):
                valid_files.append(csv_file)
                
        return valid_files
        
    def get_data_file_path(self, filename: str = None) -> str:
        """
        Get full path to a data file in datacsv folder.
        If filename is None, returns the first available valid file.
        """
        self.validate_datacsv_folder()
        
        if filename is None:
            # Return first available valid file
            valid_files = self.get_available_data_files()
            if not valid_files:
                raise ValueError("No valid data files found in datacsv folder")
            filename = valid_files[0]
            pro_log(f"Using first available data file: {filename}", tag="DataValidator")
            
        file_path = os.path.join(self.datacsv_path, filename)
        
        if not os.path.exists(file_path):
            error_msg = f"❌ CRITICAL: Data file {filename} not found in datacsv folder"
            pro_log(error_msg, level="error", tag="DataValidator")
            raise FileNotFoundError(error_msg)
            
        if not self._validate_csv_file(file_path):
            error_msg = f"❌ CRITICAL: Data file {filename} failed validation"
            pro_log(error_msg, level="error", tag="DataValidator")
            raise ValueError(error_msg)
            
        pro_log(f"✅ Validated data file path: {file_path}", tag="DataValidator")
        return file_path
        
    def load_real_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load real data from datacsv folder with full validation.
        This is the ONLY approved way to load data in the pipeline.
        """
        file_path = self.get_data_file_path(filename)
        
        try:
            df = pd.read_csv(file_path)
            pro_log(f"✅ Loaded real data: {df.shape} from {os.path.basename(file_path)}", tag="DataValidator")
            
            # Log data summary for transparency
            pro_log(f"Data range: {len(df)} rows, columns: {list(df.columns)}", tag="DataValidator")
            
            return df
            
        except Exception as e:
            error_msg = f"❌ CRITICAL: Failed to load real data from {file_path}: {e}"
            pro_log(error_msg, level="error", tag="DataValidator")
            raise ValueError(error_msg)


def prevent_dummy_data_creation(func):
    """
    Decorator to prevent functions from creating or using dummy data.
    Ensures all data operations use only real data from datacsv.
    """
    def wrapper(*args, **kwargs):
        # Check if we're trying to create dummy data
        if 'dummy' in str(kwargs).lower() or 'synthetic' in str(kwargs).lower():
            error_msg = "❌ CRITICAL: Dummy/synthetic data creation is prohibited. Only real data from datacsv allowed."
            pro_log(error_msg, level="error", tag="DataValidator")
            raise ValueError(error_msg)
            
        return func(*args, **kwargs)
    return wrapper


def enforce_real_data_only():
    """
    Global enforcement function to be called at pipeline start.
    Validates that datacsv exists and contains real data.
    """
    validator = RealDataValidator()
    validator.validate_datacsv_folder()
    
    pro_log("🛡️ Real data enforcement activated - only datacsv data allowed", tag="DataValidator")
    return validator
