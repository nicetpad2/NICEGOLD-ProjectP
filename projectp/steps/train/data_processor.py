"""
Data Processor Module for Training Pipeline

This module handles:
1. Data loading and validation
2. Data cleaning and preprocessing
3. Type conversions
4. Target value fixes
5. Data splitting
"""

import os
import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from projectp.pro_log import pro_log
from projectp.steps.backtest import load_and_prepare_main_csv
from fix_target_values import fix_target_values
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

# Global exception handler for imports
def safe_import(module_name, fallback_value=None, fallback_message=None):
    """Safely import modules with fallbacks"""
    try:
        parts = module_name.split('.')
        module = __import__(module_name)
        for part in parts[1:]:
            module = getattr(module, part)
        return module
    except ImportError as e:
        if fallback_message:
            print(f"⚠️ {fallback_message}")
        else:
            print(f"⚠️ Failed to import {module_name}, using fallback")
        return fallback_value


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing for training pipeline"""
    
    def __init__(self):
        self.features = []
        self.target = "target"
        self.datetime_columns = ["target", "Date", "datetime", "Timestamp", "Time", "date", "time", "index"]
        
    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load training data from various sources"""
        with Progress(
            SpinnerColumn(), 
            BarColumn(), 
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Loading data...", total=100)
            
            # Try loading from different sources in order of preference
            data_sources = [
                ("output_default/auto_features.parquet", "Auto-generated features"),
                ("data/raw/your_data_file.csv", "Raw CSV data"),
                ("output_default/preprocessed_super.parquet", "Preprocessed data")
            ]
            
            df = None
            source_info = {}
            
            for data_path, description in data_sources:
                if os.path.exists(data_path):
                    try:
                        if data_path.endswith('.parquet'):
                            df = pd.read_parquet(data_path)
                        else:
                            df = pd.read_csv(data_path)
                        
                        source_info = {
                            'path': data_path,
                            'description': description,
                            'size': df.shape,
                            'memory_mb': df.memory_usage(deep=True).sum() / 1e6
                        }
                        
                        pro_log(f"[DataProcessor] Loaded {description}: {df.shape}", tag="Data")
                        progress.update(task, advance=50, description=f"[green]Loaded {description}")
                        break
                        
                    except Exception as e:
                        pro_log(f"[DataProcessor] Failed to load {data_path}: {e}", level="warn", tag="Data")
                        continue
            
            if df is None:
                raise FileNotFoundError("No suitable training data found")
            
            # Basic data validation
            self._validate_data(df)
            progress.update(task, advance=50, description="[green]Data validation complete")
            
            return df, source_info
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate loaded data"""
        if df.empty:
            raise ValueError("Loaded data is empty")
        
        if self.target_col not in df.columns:
            # Try to find target column
            target_candidates = ['target', 'label', 'y', 'signal', 'trade_signal']
            for candidate in target_candidates:
                if candidate in df.columns:
                    self.target_col = candidate
                    break
            else:
                raise ValueError(f"Target column not found. Candidates: {target_candidates}")
        
        pro_log(f"[DataProcessor] Data validation passed. Target: {self.target_col}", tag="Data")
    
    def fix_target_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix target values to ensure binary classification"""
        try:
            # Import production fixer
            from ultimate_production_fix import UltimateProductionFixer
            fixer = UltimateProductionFixer()
            df = fixer.fix_target_values_ultimate(df, target_col=self.target_col)
        except ImportError:
            # Fallback to basic fixing
            pass
        
        # Ensure binary values
        unique_targets = df[self.target_col].unique()
        pro_log(f"[DataProcessor] Original target values: {sorted(unique_targets)}", tag="Data")
        
        # Convert to binary if needed
        invalid_targets = [t for t in unique_targets if t not in [0, 1]]
        if invalid_targets:
            pro_log(f"[DataProcessor] Converting invalid targets: {invalid_targets}", level="warn", tag="Data")
            df[self.target_col] = df[self.target_col].apply(lambda x: 1 if float(x) > 0 else 0)
        
        final_targets = sorted(df[self.target_col].unique())
        pro_log(f"[DataProcessor] Final target values: {final_targets}", tag="Data")
        
        return df
    
    def clean_and_convert_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Clean data and convert types for ML compatibility"""
        df_clean = df.copy()
        features = []
        
        pro_log(f"[DataProcessor] Original data types: {df_clean.dtypes.value_counts().to_dict()}", tag="Data")
        
        for col in df_clean.columns:
            try:
                # Skip target and datetime columns
                if col.lower() in [dcol.lower() for dcol in self.datetime_columns]:
                    continue
                
                # Handle object columns
                if df_clean[col].dtype == "object":
                    self._convert_object_column(df_clean, col, features)
                elif df_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    features.append(col)
                    
            except Exception as e:
                pro_log(f"[DataProcessor] Error processing column '{col}': {e}", level="warn", tag="Data")
                continue
        
        pro_log(f"[DataProcessor] Selected {len(features)} features after type conversion", tag="Data")
        return df_clean, features
    
    def _convert_object_column(self, df: pd.DataFrame, col: str, features: List[str]) -> None:
        """Convert object column to numeric if possible"""
        try:
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                return
            
            sample_val = str(non_null_values.iloc[0])
            
            # Check if it's a datetime string
            if any(char in sample_val for char in ['-', ':', '/', ' ']) and len(sample_val) > 8:
                # Convert datetime string to timestamp
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].astype('int64', errors='ignore') // 10**9
                features.append(col)
                pro_log(f"[DataProcessor] Converted datetime column '{col}' to timestamp", tag="Data")
            else:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].dtype in ['float64', 'int64']:
                    features.append(col)
                    pro_log(f"[DataProcessor] Converted object column '{col}' to numeric", tag="Data")
                    
        except Exception as e:
            pro_log(f"[DataProcessor] Failed to convert object column '{col}': {e}", level="warn", tag="Data")
    
    def split_data(self, df: pd.DataFrame, features: List[str], test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
        
        X_train_cols = features
        X_test_cols = features
        
        pro_log(f"[DataProcessor] Train split: {train_df.shape}, Test split: {test_df.shape}", tag="Data")
        
        return train_df, test_df, X_train_cols, X_test_cols
    
    def prepare_features_target(self, train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare feature matrices and target vectors"""
        X_train = train_df[features]
        y_train = train_df[self.target_col]
        X_test = test_df[features]
        y_test = test_df[self.target_col]
        
        # Fill NaN values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Convert int columns to float for MLflow compatibility
        for col in X_train.select_dtypes(include='int').columns:
            X_train[col] = X_train[col].astype('float')
            X_test[col] = X_test[col].astype('float')
        
        pro_log(f"[DataProcessor] Features prepared: {X_train.shape}", tag="Data")
        
        return X_train, y_train, X_test, y_test
