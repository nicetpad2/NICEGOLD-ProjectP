

# Assuming the refactored function is in src.model_training
# tests/test_training_fix.py
from src.model_training import train_and_export_meta_model
from unittest.mock import patch, MagicMock
import numpy as np
import os
import pandas as pd
import pytest
@patch('src.model_training.logging')
@patch('src.model_training.safe_load_csv_auto')
@patch('src.model_training.load_final_m1_data')
@patch('src.model_training.CatBoostClassifier')
@patch('src.model_training.os.path.isdir', return_value = True)
@patch('builtins.open')
def test_train_with_none_param_does_not_raise_type_error(mock_open, mock_isdir, mock_catboost, mock_load_m1, mock_load_log, mock_logging):
    """
    Test that train_and_export_meta_model does not raise a TypeError
    when a parameter that gets formatted in a log is None.
    This test simulates a scenario that would cause the "unsupported format string" error.
    """
    # Mock file loading to return valid dataframes with enough samples for splitting
    mock_load_log.return_value = pd.DataFrame({
        'entry_time': pd.to_datetime([f'2023 - 01 - {i:02}' for i in range(1, 11)]), 
        'exit_reason': ['TP'] * 10
    })
    mock_load_m1.return_value = pd.DataFrame({
        'datetime': pd.to_datetime([f'2023 - 01 - {i:02}' for i in range(1, 11)]), 
        'feature1': np.random.rand(10), 
        'signal': np.random.randint(0, 2, 10)
    })
    mock_catboost.return_value.fit.return_value = None # Mock model fitting
    # Mock predict and predict_proba to return valid numpy arrays to avoid RecursionError
    mock_catboost.return_value.predict.return_value = np.array([0, 1])
    mock_catboost.return_value.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])

    try:
        # Call the function with a None parameter that is formatted in logs
        train_and_export_meta_model(
            trade_log_path = "dummy.csv", 
            m1_data_path = "dummy.csv", 
            output_dir = "dummy_output", 
            model_purpose = 'test_none_fix', 
            shap_importance_threshold = None,  # This is the crucial part
            enable_optuna_tuning = False, 
            enable_dynamic_feature_selection = False
        )
    except TypeError as e:
        if "unsupported format string" in str(e):
            pytest.fail(f"The fix for the TypeError was not effective. Error: {e}")
        else:
            # Re - raise if it's a different TypeError
            raise e
    except Exception as e:
        pytest.fail(f"An unexpected exception occurred during the test: {e}")