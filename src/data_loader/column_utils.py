from typing import Tuple, Dict, Any, Optional, List, Union
# Import with fallback to avoid circular imports
try:
    from projectp.utils_feature import map_standard_columns, assert_no_lowercase_columns
except ImportError:
    # Fallback functions when circular import occurs
    def map_standard_columns(df):
        """Fallback function when utils_feature is not available due to circular import"""
        return df
    
    def assert_no_lowercase_columns(df):
        """Fallback function when utils_feature is not available due to circular import"""
        pass
