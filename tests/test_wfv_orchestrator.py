

import pandas as pd
import pytest
import wfv_orchestrator as wfv_orch
def test_orchestrate_walk_forward_dynamic():
    df = pd.DataFrame({'Close': range(4)})
    splits = list(wfv_orch.orchestrate_walk_forward(df, n_splits = 5))
    # dataset length 4 => max_splits = 2
    assert len(splits) == 2


def test_orchestrate_walk_forward_import_error(monkeypatch):
    monkeypatch.setattr(wfv_orch, 'TimeSeriesSplit', None)
    with pytest.raises(ImportError):
        list(wfv_orch.orchestrate_walk_forward(pd.DataFrame({'a':[1, 2, 3]})))


def test_orchestrate_walk_forward_index_slice():
    df = pd.DataFrame({'Close': range(6)}, index = [2, 1, 3, 4, 0, 5])
    splits = list(wfv_orch.orchestrate_walk_forward(df, n_splits = 2))
    # After sorting, first fold ends at index 3 and second fold at 5
    assert splits[0][1].index[ - 1] == 3
    assert splits[0][0].index.max() <= 3
    assert splits[1][1].index[ - 1] == 5