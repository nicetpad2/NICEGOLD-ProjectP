
#             f.write(f"{feat}\n")
#         'ma5': [2, 2], 'ma10': [2, 2], 'target': [1, 0]
#         'Open': [1, 2], 'High': [2, 3], 'Low': [1, 1], 'Close': [2, 2], 'Volume': [100, 200], 
#         assert False, f"run_predict failed: {e}"
#         for feat in features:
#         run_predict()
#     # Create train_features.txt with all features
#     # Mock config and data
#     # Patch open for train_features.txt
#     # Patch os.path.exists to simulate file presence
#     # Patch pd.read_csv to load our test data
#     # Remove 'Volume' to simulate missing feature
#     # Run predict (should not error, should fill missing 'Volume' with NaN)
#     # Save to CSV
#     data_path = tmp_path / 'test.csv'
#     df = df.drop(columns = ['Volume'])
#     df = pd.DataFrame({
#     df.to_csv(data_path, index = False)
#     except Exception as e:
#     features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ma5', 'ma10']
#     features_path = tmp_path / 'train_features.txt'
#     monkeypatch.setattr('builtins.open', lambda p, *a, **k: open(features_path, *a, **k))
#     monkeypatch.setattr('os.path.exists', lambda p: True)
#     monkeypatch.setattr('pandas.read_csv', lambda p, *a, **k: pd.read_csv(data_path))
#     try:
#     with open(features_path, 'w', encoding = 'utf - 8') as f:
#     })
# def test_feature_consistency_and_missing(monkeypatch, tmp_path):
# from projectp.steps.predict import run_predict
# import pandas as pd