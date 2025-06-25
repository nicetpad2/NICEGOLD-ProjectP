

#             return io.StringIO(yaml_text)
#         called['check'] = check
#         called['cmd'] = cmd
#         if file == 'config/logger_config.yaml':
#         pipeline.run_sweep(PipelineConfig(), runner = fake_run)
#         raise subprocess.CalledProcessError(1, cmd)
#         return builtins.open(file, mode, *args, **kwargs)
#     assert 'threshold_optimization.py' in called['cmd'][1]
#     assert called['check']
#     assert captured['level'] == 'DEBUG'
#     called = {}
#     captured = {}
#     def fake_open(file, mode = 'r', *args, **kwargs):
#     def fake_run(cmd, check):
#     def fake_run(cmd, check):
#     monkeypatch.setattr(builtins, 'open', fake_open)
#     monkeypatch.setattr(pipeline.logging.config, 'dictConfig', lambda cfg: captured.setdefault('level', cfg['root']['level']))
#     pipeline.run_threshold(PipelineConfig(), runner = fake_run)
#     pipeline.setup_logging('debug')
#     with pytest.raises(pipeline.PipelineError):
#     yaml_cfg = {'version': 1, 'root': {'level': 'INFO', 'handlers': []}}
#     yaml_text = yaml.dump(yaml_cfg)
# def test_run_sweep_failure():
# def test_run_threshold_success():
# def test_setup_logging_override_level(monkeypatch):
# from src.utils.pipeline_config import PipelineConfig
# import builtins
# import io
# import main as pipeline
# import os
# import pytest
# import subprocess
# import sys
# import yaml
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, ROOT_DIR)