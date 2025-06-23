from projectp.steps import backtest

def test_backtest_missing_preprocessed(tmp_path, monkeypatch):
    # config with missing preprocessed.csv
    config = {}
    result = backtest.run_backtest(config)
    assert result is None
