"""
Import compatibility manager
Handles all import issues and provides fallbacks for missing dependencies
"""

import logging
import warnings
from typing import Any, Dict, Optional, List

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

class ImportManager:
    """Manages imports with fallbacks"""
    
    def __init__(self):
        self.available_modules = {}
        self.fallback_functions = {}
        self._check_all_imports()
    
    def _check_all_imports(self):
        """Check availability of all required modules"""
        
        # Check Evidently
        try:
            from evidently.metrics import ValueDrift
            self.available_modules['evidently'] = True
            logger.info("✅ Evidently available")
        except ImportError:
            self.available_modules['evidently'] = False
            logger.warning("⚠️ Evidently not available - using fallback")
        
        # Check Pydantic SecretField
        try:
            from src.pydantic_v2_compat import SecretField
            self.available_modules['pydantic_secretfield'] = True
            logger.info("✅ Pydantic SecretField available")
        except ImportError:
            try:
                from pydantic import Field
                self.available_modules['pydantic_secretfield'] = False
                self.available_modules['pydantic_field'] = True
                logger.info("✅ Pydantic Field available (SecretField not found)")
            except ImportError:
                self.available_modules['pydantic_secretfield'] = False
                self.available_modules['pydantic_field'] = False
                logger.warning("⚠️ Pydantic not fully available - using fallback")
        
        # Check sklearn mutual_info_regression
        try:
            from sklearn.metrics import mutual_info_regression
            self.available_modules['sklearn_mutual_info'] = True
            logger.info("✅ sklearn mutual_info_regression available")
        except ImportError:
            try:
                from sklearn.feature_selection import mutual_info_regression
                self.available_modules['sklearn_mutual_info'] = True
                logger.info("✅ sklearn mutual_info_regression available (from feature_selection)")
            except ImportError:
                self.available_modules['sklearn_mutual_info'] = False
                logger.warning("⚠️ sklearn mutual_info_regression not available")
        
        # Check pipeline functions
        try:
            from src.data_loader.csv_loader import safe_load_csv_auto
            self.available_modules['csv_loader'] = True
            logger.info("✅ CSV loader available")
        except ImportError:
            self.available_modules['csv_loader'] = False
            logger.warning("⚠️ CSV loader not available - using fallback")
    
    def get_evidently_metrics(self):
        """Get Evidently metrics with fallback"""
        if self.available_modules.get('evidently', False):
            try:
                from evidently.metrics import ValueDrift, DataDrift
                return {'ValueDrift': ValueDrift, 'DataDrift': DataDrift}
            except ImportError:
                pass
        
        # Use fallback
        from src.evidently_compat import FallbackValueDrift, FallbackDataDrift
        return {'ValueDrift': FallbackValueDrift, 'DataDrift': FallbackDataDrift}
    
    def get_pydantic_field(self):
        """Get Pydantic field with fallback"""
        if self.available_modules.get('pydantic_secretfield', False):
            try:
                from src.pydantic_v2_compat import SecretField
                return SecretField
            except ImportError:
                pass
        
        if self.available_modules.get('pydantic_field', False):
            try:
                from pydantic import Field
                return Field
            except ImportError:
                pass
        
        # Use fallback
        from src.pydantic_compat import SecretField
        return SecretField
    
    def get_mutual_info_regression(self):
        """Get mutual_info_regression with fallback"""
        if self.available_modules.get('sklearn_mutual_info', False):
            try:
                from sklearn.metrics import mutual_info_regression
                return mutual_info_regression
            except ImportError:
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    return mutual_info_regression
                except ImportError:
                    pass
        
        # Fallback function
        def mutual_info_fallback(X, y, **kwargs):
            """Fallback for mutual_info_regression"""
            import numpy as np
            logger.warning("Using fallback mutual_info_regression")
            return np.random.random(X.shape[1])  # Random scores
        
        return mutual_info_fallback
    
    def get_csv_loader(self):
        """Get CSV loader with fallback"""
        if self.available_modules.get('csv_loader', False):
            try:
                from src.data_loader.csv_loader import safe_load_csv_auto
                return safe_load_csv_auto
            except ImportError:
                pass
        
        # Use fallback
        from src.pipeline_fallbacks import safe_load_csv_auto_fallback
        return safe_load_csv_auto_fallback
    
    def get_enterprise_tracker(self):
        """Get EnterpriseTracker (now available in tracking.py)"""
        try:
            from tracking import EnterpriseTracker
            return EnterpriseTracker
        except ImportError:
            # Create minimal fallback tracker
            class FallbackTracker:
                def __init__(self, *args, **kwargs):
                    logger.warning("Using fallback tracker")
                
                def track_experiment(self, *args, **kwargs):
                    return self
                
                def log_params(self, *args, **kwargs):
                    pass
                
                def log_metrics(self, *args, **kwargs):
                    pass
                
                def __enter__(self):
                    return self
                
                def __exit__(self, *args):
                    pass
            
            return FallbackTracker

# Global import manager instance
import_manager = ImportManager()

# Convenience functions for easy access
def get_evidently_valuedrift():
    """Get ValueDrift class"""
    metrics = import_manager.get_evidently_metrics()
    return metrics['ValueDrift']

def get_pydantic_secretfield():
    """Get SecretField class"""
    return import_manager.get_pydantic_field()

def get_mutual_info_regression():
    """Get mutual_info_regression function"""
    return import_manager.get_mutual_info_regression()

def get_safe_load_csv_auto():
    """Get safe_load_csv_auto function"""
    return import_manager.get_csv_loader()

def get_enterprise_tracker():
    """Get EnterpriseTracker class"""
    return import_manager.get_enterprise_tracker()

def log_import_status():
    """Log the status of all imports"""
    logger.info("=== Import Status ===")
    for module, available in import_manager.available_modules.items():
        status = "✅ Available" if available else "⚠️ Using fallback"
        logger.info(f"{module}: {status}")
    logger.info("==================")

# Log status on import
log_import_status()
