import pandas as pd
import sys
from projectp.utils_feature import map_standard_columns, assert_no_lowercase_columns

def validate_column_names(file_path):
    df = pd.read_parquet(file_path)
    df = map_standard_columns(df)
    try:
        assert_no_lowercase_columns(df)
        print(f"[PASS] {file_path}: No lowercase columns remain after mapping.")
    except AssertionError as e:
        print(f"[FAIL] {file_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_column_names.py <parquet_file>")
        sys.exit(1)
    validate_column_names(sys.argv[1])
