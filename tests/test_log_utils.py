from src.utils import log_utils

def test_set_log_context_and_clear():
    log_utils.set_log_context(user = "test", run_id = "abc123")
    assert log_utils.LOG_CONTEXT["user"] == "test"
    assert log_utils.LOG_CONTEXT["run_id"] == "abc123"
    log_utils.clear_log_context()
    assert log_utils.LOG_CONTEXT == {}

def test_pro_log_and_json(tmp_path):
    log_utils.set_log_context(user = "testuser")
    log_utils.pro_log("test message", tag = "Test", level = "INFO", extra = "value")
    log_utils.pro_log_json({"event": "test_event"}, tag = "TestJson", level = "INFO")
    # Export log to a temp file
    out = tmp_path / "log.jsonl"
    log_utils.export_log_to(str(out), level = "INFO")
    assert out.exists()