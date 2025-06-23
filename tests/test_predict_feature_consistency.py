# import pandas as pd
# from projectp.steps.predict import run_predict

# def test_feature_consistency_and_missing(monkeypatch, tmp_path):
#     # Mock config and data
#     df = pd.DataFrame({
#         'Open': [1,2], 'High': [2,3], 'Low': [1,1], 'Close': [2,2], 'Volume': [100,200],
#         'ma5': [2,2], 'ma10': [2,2], 'target': [1,0]
#     })
#     # Remove 'Volume' to simulate missing feature
#     df = df.drop(columns=['Volume'])
#     # Save to CSV
#     data_path = tmp_path / 'test.csv'
#     df.to_csv(data_path, index=False)
#     # Create train_features.txt with all features
#     features = ['Open','High','Low','Close','Volume','ma5','ma10']
#     features_path = tmp_path / 'train_features.txt'
#     with open(features_path, 'w', encoding='utf-8') as f:
#         for feat in features:
#             f.write(f"{feat}\n")
#     # Patch os.path.exists to simulate file presence
#     monkeypatch.setattr('os.path.exists', lambda p: True)
#     # Patch pd.read_csv to load our test data
#     monkeypatch.setattr('pandas.read_csv', lambda p, *a, **k: pd.read_csv(data_path))
#     # Patch open for train_features.txt
#     monkeypatch.setattr('builtins.open', lambda p, *a, **k: open(features_path, *a, **k))
#     # Run predict (should not error, should fill missing 'Volume' with NaN)
#     try:
#         run_predict()
#     except Exception as e:
#         assert False, f"run_predict failed: {e}"
