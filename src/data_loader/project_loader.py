from .csv_loader import safe_load_csv_auto
import os
import pandas as pd
from projectp.utils_feature import map_standard_columns, assert_no_lowercase_columns

def load_project_csvs(row_limit=None, clean=False):
    # Example: Load all project CSVs in a folder
    folder = 'data/'
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    dfs = {}
    for f in csv_files:
        df = safe_load_csv_auto(os.path.join(folder, f), row_limit=row_limit)
        if clean:
            df = df.dropna()
        dfs[f] = df
    return dfs

def get_base_csv_data(row_limit=None, force_reload=False):
    # Example: Load a base CSV file
    base_path = 'data/base.csv'
    df = safe_load_csv_auto(base_path, row_limit=row_limit)
    return df

def load_raw_data_m1(path=None):
    if path is None:
        path = 'data/XAUUSD_M1.csv'
    df = safe_load_csv_auto(path)
    return df

def load_raw_data_m15(path=None):
    if path is None:
        path = 'data/XAUUSD_M15.csv'
    df = safe_load_csv_auto(path)
    return df
