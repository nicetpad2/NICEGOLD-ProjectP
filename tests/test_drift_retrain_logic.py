#
#
#
#     assert not obs.needs_retrain(0, threshold = 0.5)
#     assert obs.needs_retrain(0, threshold = 0.5)
#     df_test = pd.DataFrame({'ADX': [0.1] * 20})
#     df_test = pd.DataFrame({'ADX': [2.0] * 20})
#     df_train = pd.DataFrame({'ADX': [0.0] * 20})
#     df_train = pd.DataFrame({'ADX': [0.0] * 20})
#     obs = DriftObserver(['ADX'])
#     obs = DriftObserver(['ADX'])
#     obs.analyze_fold(df_train, df_test, 0)
#     obs.analyze_fold(df_train, df_test, 0)
# def test_needs_retrain_false():
# def test_needs_retrain_true():
# from src.strategy.drift import DriftObserver
# import pandas as pd