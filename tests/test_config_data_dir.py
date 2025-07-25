

from src import config as cfg
import importlib
import pytest
import sys
def test_config_data_dir(tmp_path, monkeypatch):
    """Ensure DATA_DIR is isolated using pytest tmp_path."""
    # [Patch] Use pytest tmp_path for isolation
    cfg.DATA_DIR = tmp_path / "data"
    cfg.DATA_DIR.mkdir(parents = True, exist_ok = True)

    monkeypatch.delenv('SYMBOL', raising = False)
    monkeypatch.delenv('TIMEFRAME', raising = False)
    importlib.reload(cfg)

    assert cfg.DATA_DIR.is_dir()
    assert cfg.SYMBOL == 'XAUUSD'
    assert cfg.TIMEFRAME == 'M1'
    for attr in [
        'subsample', 
        'colsample_bylevel', 
        'bagging_temperature', 
        'random_strength', 
        'seed', 
    ]:
        assert hasattr(cfg, attr)
        assert getattr(cfg, attr) is None