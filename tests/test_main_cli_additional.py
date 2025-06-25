#
#
#
#
#
#
#
#
#
#         pipeline.run_backtest_pipeline(pd.DataFrame(), pd.DataFrame(), 'm.joblib', 0.1)
#         raise RuntimeError('fail')
#     assert any('GPU detected' in m for m in msgs)
#     assert any('Internal backtest error' in m for m in logs)
#     assert called['stage'] == 'backtest'
#     assert res == 0
#     called = {}
#     def boom(stage):
#     import src.main as src_main
#     import src.main as src_main
#     logs = []
#     monkeypatch.setattr(pipeline, 'has_gpu', lambda: True)
#     monkeypatch.setattr(pipeline, 'run_report', lambda c: None)
#     monkeypatch.setattr(pipeline, 'setup_logging', lambda level: None)
#     monkeypatch.setattr(pipeline.logger, 'exception', lambda msg, *a, **k: logs.append(msg))
#     monkeypatch.setattr(pipeline.logger, 'info', lambda msg, *a, **k: msgs.append(msg % a))
#     monkeypatch.setattr(src_main, 'run_pipeline_stage', boom)
#     monkeypatch.setattr(src_main, 'run_pipeline_stage', lambda s: called.setdefault('stage', s))
#     msgs = []
#     pipeline.run_backtest_pipeline(pd.DataFrame(), pd.DataFrame(), 'm.joblib', 0.1)
#     res = pipeline.main([' -  - mode', 'report'])
#     with pytest.raises(RuntimeError):
# def test_main_report_gpu(monkeypatch):
# def test_run_backtest_pipeline_exception(monkeypatch):
# def test_run_backtest_pipeline_success(monkeypatch):
# import logging
# import main as pipeline
# import os
# import pandas as pd
# import pytest
# import sys
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, ROOT_DIR)