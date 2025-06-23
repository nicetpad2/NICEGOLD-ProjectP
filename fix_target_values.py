"""
Fix target values for ML model training.
This module helps convert the -1 values in the target column to values
that can be handled by scikit-learn classifiers.
"""

import pandas as pd
import os
import warnings
from projectp.pro_log import pro_log

def fix_target_values(df, target_col='target'):
    """
    Fix target values by converting -1 to 2 or filtering out rows with -1.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe with target column
    target_col : str, optional
        The name of the target column, by default 'target'
        
    Returns
    -------
    pd.DataFrame
        The dataframe with fixed target values
    
    Notes
    -----
    This handles the "Unknown class label: '-1'" error that occurs
    when training ML models with scikit-learn.
    """
    if target_col not in df.columns:
        warnings.warn(f"Target column '{target_col}' not found in dataframe")
        return df
    
    # Check if we have -1 values
    has_neg_one = (df[target_col] == -1).any()
    
    if not has_neg_one:
        return df
    
    # Method 1: Convert to binary classification (0, 1) - PRODUCTION READY
    df_fixed = df.copy()
    
    # Get current distribution
    original_dist = df_fixed[target_col].value_counts().to_dict()
    pro_log(f"[TargetFix] Original target distribution: {original_dist}", tag="Fix", level="info")
    
    # Convert all values to binary (0 or 1)
    def to_binary(val):
        try:
            val = float(val)
            if val > 0:
                return 1
            else:
                return 0
        except:
            return 0
    
    df_fixed[target_col] = df_fixed[target_col].apply(to_binary)
    
    # Log the conversion
    final_dist = df_fixed[target_col].value_counts().to_dict()
    pro_log(f"[TargetFix] Final binary target distribution: {final_dist}", tag="Fix", level="info")
    
    # Handle extreme imbalance
    if 1 in final_dist and 0 in final_dist:
        imbalance_ratio = final_dist[0] / final_dist[1]
        if imbalance_ratio > 100:
            pro_log(f"[TargetFix] WARNING: Extreme class imbalance detected: {imbalance_ratio:.2f}:1", 
                   tag="Fix", level="warn")
        elif imbalance_ratio < 0.01:
            pro_log(f"[TargetFix] WARNING: Reverse extreme imbalance detected: {1/imbalance_ratio:.2f}:1", 
                   tag="Fix", level="warn")
    
    return df_fixed

def prepare_data_for_training(input_path=None):
    """
    Prepare dataframe for model training by fixing target values.
    
    Parameters
    ----------
    input_path : str, optional
        Path to input CSV or Parquet file, by default None
        If None, defaults to 'output_default/preprocessed.csv'
        
    Returns
    -------
    pd.DataFrame
        The prepared dataframe ready for model training
    """
    if input_path is None:
        input_path = os.path.join('output_default', 'preprocessed.csv')
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load the data
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    # Fix target values
    df_fixed = fix_target_values(df)
    
    # Save the fixed data
    output_path = os.path.join('output_default', 'preprocessed_fixed.csv')
    df_fixed.to_csv(output_path, index=False)
    pro_log(f"[TargetFix] Saved fixed data to: {output_path}", tag="Fix", level="info")
    
    return df_fixed

if __name__ == "__main__":
    # Simple test to make sure the function works
    df_test = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'target': [0, 1, -1, 0, 1]
    })
    
    df_fixed = fix_target_values(df_test)
    print("Original target values:", df_test['target'].unique())
    print("Fixed target values:", df_fixed['target'].unique())
