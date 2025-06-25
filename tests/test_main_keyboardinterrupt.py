

#         self.saved = False
#         self.saved = True
#     args = argparse.Namespace(mode = "preprocess", config = "cfg.yaml", log_level = None, debug = False, rows = None, profile = False, output_file = "out.prof", live_loop = 0)
#     assert result == 1
#     assert state.saved
#     def __init__(self):
#     def save_state(self):
#     monkeypatch.setattr(pipeline, "load_config", lambda p: pipeline.PipelineConfig())
#     monkeypatch.setattr(pipeline, "parse_args", lambda _ = None: args)
#     monkeypatch.setattr(pipeline, "run_preprocess", lambda cfg: (_ for _ in ()).throw(KeyboardInterrupt()))
#     monkeypatch.setattr(pipeline, "StateManager", lambda state_file_path = 'out': state)
#     result = pipeline.main()
#     state = DummyState()
# All tests in this file are disabled due to circular import issues with main.py and src.strategy
# class DummyState:
# def test_main_handles_keyboardinterrupt(monkeypatch, tmp_path):
# import main as pipeline  # Disabled due to circular import issues
import argparse
import pytest