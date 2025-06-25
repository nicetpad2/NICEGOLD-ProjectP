

#         main.run_preprocess(main.PipelineConfig(), runner = subprocess.run)
#         raise subprocess.CalledProcessError(1, 'cmd')
#     def raise_error(*args, **kwargs):
#     monkeypatch.setattr(subprocess, 'run', raise_error)
#     with pytest.raises(PipelineError):
# def test_run_preprocess_error(monkeypatch):
# from src.utils.errors import PipelineError
# import main
# import pytest
# import subprocess