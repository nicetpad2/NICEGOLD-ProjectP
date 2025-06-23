# import os
# import threading
# import time
# from backtest_engine import log_resource_usage

# def test_resource_monitor_creates_log(tmp_path):
#     log_path = tmp_path / "resource_usage.log"
#     stop_event = threading.Event()
#     t = threading.Thread(target=log_resource_usage, args=(1, str(log_path), stop_event))
#     t.start()
#     time.sleep(2)
#     stop_event.set()
#     t.join()
#     assert os.path.exists(log_path)
#     with open(log_path) as f:
#         lines = f.readlines()
#     assert len(lines) > 1  # header + at least one record
