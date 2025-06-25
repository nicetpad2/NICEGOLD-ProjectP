

#         'atr_at_entry': 1.0, 
#         'entry_bar_count': 0, 
#         'entry_price': 10.0, 
#         'entry_price': 10.0, 
#         'entry_time': '2023 - 01 - 01'
#         'entry_time': '2023 - 01 - 01'
#         'original_sl_price': 9.8, 
#         'side': 'BUY', 
#         'side': 'BUY', 
#         'sl_price': 9.8, 
#         'sl_price': 9.8, 
#         'tp_price': 10.2, 
#         be_sl_counter = 0, tsl_counter = 0
#         fold_sl_multiplier_base = 2.0, base_tp_multiplier_config = 2.0, 
#         now = pd.Timestamp('2023 - 01 - 01 00:01'), base_be_r_thresh = 1.0, 
#         order, current_high = 11.0, current_low = 9.9, current_atr = 1.0, avg_atr = 1.0, 
#     )
#     assert be and order['be_triggered'] and order['sl_price'] == 10.0 and be_count == 1
#     assert closed and reason == 'BE - SL'
#     assert closed and reason == 'SL' and price == 9.8
#     assert closed and reason == 'TP' and price == 10.2
#     closed, price, reason, _ = strategy.check_main_exit_conditions(order, row_be, 2, pd.Timestamp('2023 - 01 - 01 00:02'))
#     closed, price, reason, _ = strategy.check_main_exit_conditions(order, row_sl, 1, pd.Timestamp('2023 - 01 - 01 00:01'))
#     closed, price, reason, _ = strategy.check_main_exit_conditions(order, row_tp, 3, pd.Timestamp('2023 - 01 - 01 00:03'))
#     monkeypatch.setattr(strategy, 'dynamic_tp2_multiplier', lambda a, b, base = None: 2.0)
#     monkeypatch.setattr(strategy, 'update_trailing_tp2', lambda order, atr, mult: order)
#     monkeypatch.setattr(strategy, 'update_tsl_only', lambda *a, **k: (a[0], False))
#     order = {
#     order = {
#     order, be, tsl, be_count, tsl_count = strategy._update_open_order_state(
#     order['be_triggered'] = False
#     order['be_triggered'] = True
#     order['sl_price'] = 10.0
#     order['sl_price'] = 9.8
#     row_be = pd.Series({'High': 10.1, 'Low': 9.9, 'Close': 10.0})
#     row_sl = pd.Series({'High': 10.1, 'Low': 9.7, 'Close': 9.8})
#     row_tp = pd.Series({'High': 10.3, 'Low': 9.9, 'Close': 10.2})
#     }
#     }
# def test_check_main_exit_conditions_sl_tp_be():
# def test_update_open_order_state_be(monkeypatch):
# from src import strategy
# import pandas as pd