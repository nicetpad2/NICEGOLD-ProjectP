#
#
#     assert res_seq.equals(res_par)
#     df = pd.DataFrame({'Open': [1, 2], 'High':[2, 3], 'Low':[1, 2], 'Close':[2, 3]})
#     params = [('XAUUSD', df)]
#     res_par = run_parallel_feature_engineering(params, processes = 1)[0]
#     res_seq = calculate_features_for_fold(params[0])
# def test_parallel_feature_engineering():
# from profile_backtest import calculate_features_for_fold, run_parallel_feature_engineering
# import pandas as pd