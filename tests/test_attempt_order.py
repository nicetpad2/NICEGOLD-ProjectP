#
#
#
#
#
#
#
#         ok, reasons = strategy.attempt_order("BUY", 1.0, {})
#         ok, reasons = strategy.attempt_order("BUY", 1.0, {})
#         ok, reasons = strategy.attempt_order("SELL", 2.0, params)
#     assert "OMS_DISABLED" in caplog.text
#     assert "Order Executed" in caplog.text
#     assert not ok
#     assert not ok and reasons == ["OMS_DISABLED"]
#     assert ok and reasons == []
#     assert reasons[0] == "OMS_DISABLED"
#     assert set(reasons) == {"OMS_DISABLED", "KILL_SWITCH_ACTIVE", "PAPER_MODE_SIMULATION"}
#     monkeypatch.setattr(strategy, "OMS_ENABLED", False)
#     monkeypatch.setattr(strategy, "OMS_ENABLED", False)
#     monkeypatch.setattr(strategy, "OMS_ENABLED", True)
#     monkeypatch.setattr(strategy, "OMS_ENABLED", True)
#     params = {"kill_switch_active": True, "paper_mode": True}
#     with caplog.at_level(logging.INFO):
#     with caplog.at_level(logging.WARNING):
#     with caplog.at_level(logging.WARNING):
# def test_attempt_order_blocked_oms(monkeypatch, caplog):
# def test_attempt_order_executed(monkeypatch, caplog):
# def test_attempt_order_multiple_reasons(monkeypatch, caplog):
# from src import strategy
# import logging
# import os
# import sys
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, ROOT_DIR)