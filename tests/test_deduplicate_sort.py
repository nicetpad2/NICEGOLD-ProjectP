
from src.data_loader import deduplicate_and_sort
import os
import pandas as pd
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_deduplicate_and_sort_basic():
    df = pd.DataFrame({
        'Date': [20240101, 20240101, 20240101], 
        'Timestamp': ['00:00:00', '00:00:00', '00:01:00'], 
        'Close': [1, 2, 3]
    })
    res = deduplicate_and_sort(df, subset_cols = ['Date', 'Timestamp'])
    assert len(res) == 2
    assert list(res['Timestamp']) == ['00:00:00', '00:01:00']
    assert res.iloc[0]['Close'] == 2


def test_deduplicate_and_sort_missing_columns():
    df = pd.DataFrame({'A': [1, 1]})
    res = deduplicate_and_sort(df, subset_cols = ['Date', 'Timestamp'])
    assert len(res) == 2