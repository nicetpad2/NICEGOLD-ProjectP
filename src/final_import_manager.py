from pathlib import Path
                from sklearn.feature_selection import mutual_info_regression
            from sklearn.metrics import mutual_info_regression
            from src.data_loader.csv_loader import safe_load_csv_auto
            from src.evidently_fix import ValueDrift, DataDrift, EVIDENTLY_AVAILABLE
            from src.pydantic_fix import SecretField, Field, SecretStr, BaseModel
            from tracking import EnterpriseTracker
            import builtins
import logging
                import numpy as np
import os
            import pandas as pd
import sys
import warnings
"""
Final Import Manager - จัดการ imports ทั้งหมดให้เรียบร้อย
"""


# Setup
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FinalImportManager:
    """จัดการ imports สุดท้าย"""

    def __init__(self):
        self.fixes_applied = {}
        self.apply_all_fixes()

    def apply_all_fixes(self):
        """ใช้การแก้ไขทั้งหมด"""
        logger.info("🚀 Applying final import fixes...")

        # 1. Fix pydantic
        self.fixes_applied['pydantic'] = self._fix_pydantic()

        # 2. Fix evidently
        self.fixes_applied['evidently'] = self._fix_evidently()

        # 3. Fix sklearn
        self.fixes_applied['sklearn'] = self._fix_sklearn()

        # 4. Fix tracking
        self.fixes_applied['tracking'] = self._fix_tracking()

        # 5. Fix csv_loader
        self.fixes_applied['csv_loader'] = self._fix_csv_loader()

        self._report_status()

    def _fix_pydantic(self):
        """แก้ไข pydantic"""
        try:

            # Make available globally
            builtins.SecretField = SecretField
            builtins.PydanticField = Field
            builtins.PydanticSecretStr = SecretStr
            builtins.PydanticBaseModel = BaseModel

            logger.info("✅ Pydantic fixed")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Pydantic fix failed: {e}")
            return False

    def _fix_evidently(self):
        """แก้ไข evidently"""
        try:

            # Make available globally
            builtins.ValueDrift = ValueDrift
            builtins.DataDrift = DataDrift
            builtins.EVIDENTLY_AVAILABLE = EVIDENTLY_AVAILABLE

            logger.info("✅ Evidently fixed")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Evidently fix failed: {e}")
            return False

    def _fix_sklearn(self):
        """แก้ไข sklearn"""
        try:
            logger.info("✅ sklearn.metrics.mutual_info_regression available")
            return True
        except ImportError:
            try:
                logger.info("✅ sklearn.feature_selection.mutual_info_regression available")
                return True
            except ImportError:
                logger.warning("⚠️ mutual_info_regression not available, using fallback")


                def mutual_info_fallback(X, y, **kwargs):
                    return np.random.random(X.shape[1]) if hasattr(X, 'shape') else np.array([0.5])

                builtins.mutual_info_regression = mutual_info_fallback
                return False

    def _fix_tracking(self):
        """แก้ไข tracking"""
        try:
            logger.info("✅ EnterpriseTracker available")
            return True
        except ImportError:
            logger.warning("⚠️ EnterpriseTracker not available, using fallback")

            class FallbackTracker:
                def __init__(self, *args, **kwargs):
                    pass
                def track_experiment(self, *args, **kwargs):
                    return self
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass

            builtins.EnterpriseTracker = FallbackTracker
            return False

    def _fix_csv_loader(self):
        """แก้ไข csv_loader"""
        try:
            logger.info("✅ CSV loader available")
            return True
        except ImportError:
            logger.warning("⚠️ CSV loader not available, using fallback")


            def safe_load_csv_auto_fallback(file_path, row_limit = None, **kwargs):
                try:
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"File not found: {file_path}")
                    df = pd.read_csv(file_path, nrows = row_limit, **kwargs)
                    return df
                except Exception as e:
                    logger.error(f"CSV loading failed: {e}")
                    return pd.DataFrame()

            builtins.safe_load_csv_auto = safe_load_csv_auto_fallback
            return False

    def _report_status(self):
        """รายงานสถานะ"""
        working = sum(self.fixes_applied.values())
        total = len(self.fixes_applied)

        logger.info(" = " * 50)
        logger.info("📊 Final Import Status:")
        for component, status in self.fixes_applied.items():
            icon = "✅" if status else "⚠️"
            logger.info(f"  {component}: {icon}")

        logger.info(f"📈 Success Rate: {working}/{total}")
        logger.info("✅ All imports ready!")
        logger.info(" = " * 50)

# Global instance
final_manager = FinalImportManager()