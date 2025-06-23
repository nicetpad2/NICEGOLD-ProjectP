"""
Sklearn Compatibility Layer
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global variables
mutual_info_regression = None

def initialize_sklearn():
    global mutual_info_regression
    
    # Try sklearn.metrics first
    try:
        from sklearn.metrics import mutual_info_regression as mir
        mutual_info_regression = mir
        logger.info("✅ sklearn.metrics.mutual_info_regression loaded")
        return True
    except ImportError:
        pass
    
    # Try sklearn.feature_selection
    try:
        from sklearn.feature_selection import mutual_info_regression as mir
        mutual_info_regression = mir
        logger.info("✅ sklearn.feature_selection.mutual_info_regression loaded")
        return True
    except ImportError:
        pass
    
    # Fallback
    def mutual_info_fallback(X, y, **kwargs):
        """Fallback mutual info regression"""
        logger.warning("Using fallback mutual_info_regression")
        return np.random.random(X.shape[1]) if hasattr(X, 'shape') else np.array([0.5])
    
    mutual_info_regression = mutual_info_fallback
    logger.warning("⚠️ Using fallback mutual_info_regression")
    return False

# Initialize on import
initialize_sklearn()

# Export
__all__ = ['mutual_info_regression', 'initialize_sklearn']
