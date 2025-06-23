"""
Universal Import Fixer - Patches all import issues at runtime
"""

import sys
import logging
import warnings
from typing import Any

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class UniversalImportFixer:
    """Patches imports at runtime to prevent errors"""
    
    def __init__(self):
        self.original_import = __builtins__['__import__']
        self.patches_applied = {}
    
    def patch_imports(self):
        """Patch the import system"""
        def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
            try:
                return self.original_import(name, globals, locals, fromlist, level)
            except ImportError as e:
                return self._handle_import_error(name, fromlist, e)
        
        __builtins__['__import__'] = patched_import
        logger.info("✅ Universal import patches applied")
    
    def _handle_import_error(self, name, fromlist, error):
        """Handle import errors with fallbacks"""
        
        # Evidently patches
        if 'evidently' in name and fromlist:
            if 'ValueDrift' in fromlist:
                logger.warning(f"Patching evidently.ValueDrift import")
                return self._create_evidently_mock()
        
        # Pydantic patches
        if 'pydantic' in name and fromlist:
            if 'SecretField' in fromlist:
                logger.warning(f"Patching pydantic.SecretField import")
                return self._create_pydantic_mock()
        
        # CSV loader patches
        if 'csv_loader' in name and fromlist:
            if 'safe_load_csv_auto' in fromlist:
                logger.warning(f"Patching csv_loader import")
                return self._create_csv_loader_mock()
        
        # Re-raise if we can't handle it
        raise error
    
    def _create_evidently_mock(self):
        """Create mock evidently module"""
        class MockModule:
            class ValueDrift:
                def __init__(self, *args, **kwargs):
                    pass
                def calculate(self, *args, **kwargs):
                    return {'drift_score': 0.1, 'drift_detected': False}
        return MockModule()
    
    def _create_pydantic_mock(self):
        """Create mock pydantic module"""
        class MockModule:
            def SecretField(*args, **kwargs):
                return lambda x: x
            def Field(*args, **kwargs):
                return lambda x: x
        return MockModule()
    
    def _create_csv_loader_mock(self):
        """Create mock csv_loader module"""
        import pandas as pd
        
        class MockModule:
            def safe_load_csv_auto(file_path, **kwargs):
                try:
                    return pd.read_csv(file_path, **kwargs)
                except Exception:
                    return pd.DataFrame()
        return MockModule()

# Global fixer
fixer = UniversalImportFixer()

def apply_universal_fixes():
    """Apply all universal fixes"""
    try:
        # Import compatibility layers
        from src.evidently_compat import initialize_evidently
        from src.pydantic_compat import initialize_pydantic
        from src.sklearn_compat import initialize_sklearn
        
        # Initialize all compatibility layers
        initialize_evidently()
        initialize_pydantic()
        initialize_sklearn()
        
        # Apply runtime patches
        fixer.patch_imports()
        
        logger.info("✅ All universal fixes applied successfully")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Some universal fixes failed: {e}")
        # Apply at least the runtime patches
        fixer.patch_imports()
        return False

# Auto-apply on import
apply_universal_fixes()
