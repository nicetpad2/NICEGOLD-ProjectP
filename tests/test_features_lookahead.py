

from src.features import rolling_zscore
import numpy as np
import pandas as pd
def test_rolling_zscore_no_lookahead():
    series = pd.Series([1, 2, 3, 4, 5], dtype = 'float32')
    full = rolling_zscore(series, 3)
    for i in range(3, len(series) + 1):
        partial = rolling_zscore(series.iloc[:i], 3)
        assert np.isclose(full.iloc[i - 1], partial.iloc[ - 1])