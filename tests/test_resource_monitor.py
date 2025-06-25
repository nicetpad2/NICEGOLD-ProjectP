
#         lines = f.readlines()
#     assert len(lines) > 1  # header + at least one record
#     assert os.path.exists(log_path)
#     log_path = tmp_path / "resource_usage.log"
#     stop_event = threading.Event()
#     stop_event.set()
#     t = threading.Thread(target = log_resource_usage, args = (1, str(log_path), stop_event))
#     t.join()
#     t.start()
#     time.sleep(2)
#     with open(log_path) as f:
# def test_resource_monitor_creates_log(tmp_path):
# from backtest_engine import log_resource_usage
# import os
# import threading
# import time