

#     assert abs(order["sl_price"] - 1.0001) < 1e - 6
#     assert order["be_triggered"]
#     assert sl < 1.0 and tp <= 1.02
#     assert triggered
#     order = {"side": "BUY", "entry_price": 1.0000, "tp1_price": 1.0100, "sl_price": 0.9900}
#     order, triggered = update_breakeven_half_tp(order, current_high = 1.0051, current_low = 0.9990, now = pd.Timestamp('2024 - 01 - 01'))
#     sl, tp = adjust_sl_tp_oms(1.0, 0.9995, 1.02, 0.005, "BUY", 10, 50)
# def test_adjust_sl_tp_oms_margin():
# def test_update_breakeven_half_tp_buy():
# from src.strategy import update_breakeven_half_tp, adjust_sl_tp_oms
# import pandas as pd