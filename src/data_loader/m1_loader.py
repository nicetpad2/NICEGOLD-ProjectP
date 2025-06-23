"""
M1 data loader module
"""
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, Any, Union, Tuple

def load_final_m1_data(path: str, validate: bool = True) -> pd.DataFrame:
    """
    Load and validate M1 data from a given path.
    
    Args:
        path: Path to the M1 data file
        validate: Whether to validate the data
        
    Returns:
        DataFrame containing the M1 data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"M1 data file not found: {path}")
    
    # Load the data
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    # Basic validation
    if validate:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df
