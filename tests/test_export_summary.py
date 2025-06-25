
import os
import pandas as pd
import sys
import tuning.hyperparameter_sweep as hs
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


def test_export_summary_adds_columns(tmp_path):
    df = pd.DataFrame({'a': [1]})
    out = tmp_path / 'summary.csv'
    hs.export_summary(df, str(out))
    loaded = pd.read_csv(out)
    assert 'metric' in loaded.columns
    assert 'best_param' in loaded.columns