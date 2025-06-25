
# pytest.skip("Disabled: expects non - existent functionality", allow_module_level = True)
from feature_engineering import add_domain_and_lagged_features, check_feature_collinearity
import pandas as pd
import pytest
def test_add_domain_and_lagged_features_empty():
    df = pd.DataFrame()
    result = add_domain_and_lagged_features(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_check_feature_collinearity_all_constant():
    df = pd.DataFrame({
        'A': [1, 1, 1, 1], 
        'B': [2, 2, 2, 2], 
        'C': [3, 3, 3, 3], 
    })
    # Should not raise error
    check_feature_collinearity(df)

def test_check_feature_collinearity_high_corr():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4], 
        'B': [2, 4, 6, 8],  # perfectly correlated with A
        'C': [1, 1, 1, 1], 
    })
    # Should not raise error, just log
    check_feature_collinearity(df)