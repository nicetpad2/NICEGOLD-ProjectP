
#         result = _resolve_close_index(df, missing_idx, missing_idx)
#     assert "not in df_sim.index" in caplog.text
#     assert result in idxs
#     df = pd.DataFrame(index = idxs)
#     idxs = pd.to_datetime(["2024 - 01 - 01 00:00:00", "2024 - 01 - 01 00:02:00"])
#     missing_idx = pd.Timestamp("2024 - 01 - 01 00:01:00")
#     with caplog.at_level('WARNING'):
# def test_resolve_close_index_uses_nearest(caplog):
# from src.strategy import _resolve_close_index
import logging
import pandas as pd