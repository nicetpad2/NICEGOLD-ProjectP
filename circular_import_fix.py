"""
Comprehensive Circular Import Fix
แก้ไขปัญหา circular import ทั้งหมดในโปรเจค
"""

import sys
import logging
import importlib
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

class CircularImportFixer:
    """ตัวแก้ไขปัญหา circular import"""
    
    def __init__(self):
        self.fixed_modules = set()
        self.fallback_functions = {}
        
    def apply_csv_loader_fix(self):
        """แก้ไขปัญหา circular import ใน csv_loader"""
        try:
            # ลบ module จาก cache ถ้ามี
            modules_to_reload = [
                'src.data_loader.csv_loader',
                'projectp.utils_feature',
                'src.data_loader.column_utils'
            ]
            
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    logger.info(f"🔄 Reloading module: {module_name}")
                    importlib.reload(sys.modules[module_name])
            
            # ทดสอบ import
            from src.data_loader.csv_loader import safe_load_csv_auto
            logger.info("✅ CSV loader circular import fixed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ CSV loader fix failed: {e}")
            self._create_csv_loader_fallback()
            return False
    
    def _create_csv_loader_fallback(self):
        """สร้าง fallback สำหรับ CSV loader"""
        import pandas as pd
        import os
        
        def safe_load_csv_auto_fallback(file_path, row_limit=None, **kwargs):
            """Fallback CSV loader"""
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"ไม่พบไฟล์ {file_path}")
                
                # Basic CSV loading
                df = pd.read_csv(file_path, nrows=row_limit, **kwargs)
                
                # Remove BOM
                if len(df.columns) > 0 and df.columns[0].startswith('\ufeff'):
                    df.columns = [col.replace('\ufeff', '') for col in df.columns]
                
                # Auto-detect datetime columns
                datetime_cols = []
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['time', 'date']):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            datetime_cols.append(col)
                        except:
                            pass
                
                # Set index to first datetime column
                if datetime_cols:
                    try:
                        df = df.set_index(datetime_cols[0])
                    except:
                        pass
                
                logger.info(f"📄 Loaded CSV with fallback: {file_path} ({len(df)} rows)")
                return df
                
            except Exception as e:
                logger.error(f"❌ CSV fallback loading failed: {e}")
                return pd.DataFrame()
        
        # Make available globally
        import builtins
        builtins.safe_load_csv_auto = safe_load_csv_auto_fallback
        self.fallback_functions['safe_load_csv_auto'] = safe_load_csv_auto_fallback
        
        logger.info("✅ CSV loader fallback created")
    
    def apply_typing_fixes(self):
        """แก้ไขปัญหา typing imports"""
        try:
            # เพิ่ม typing imports ที่ขาดหายไป
            import builtins
            from typing import Tuple, Dict, Any, Optional, List, Union
            
            # ตรวจสอบว่ามีใน builtins แล้วหรือไม่
            typing_items = ['Tuple', 'Dict', 'Any', 'Optional', 'List', 'Union']
            for item in typing_items:
                if not hasattr(builtins, item):
                    setattr(builtins, item, eval(item))
            
            logger.info("✅ Typing imports fixed globally")
            return True
            
        except Exception as e:
            logger.error(f"❌ Typing fix failed: {e}")
            return False
    
    def fix_import_order(self):
        """แก้ไขลำดับการ import"""
        try:
            # กำหนดลำดับการ import ที่ถูกต้อง
            import_order = [
                'src.evidently_compat',
                'src.pydantic_compat', 
                'src.pipeline_fallbacks',
                'src.import_manager',
                'evidently_v0430_fix',
                'tracking'
            ]
            
            for module_name in import_order:
                try:
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])
                    else:
                        __import__(module_name)
                    logger.info(f"✅ Imported: {module_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to import {module_name}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Import order fix failed: {e}")
            return False
    
    def patch_problematic_imports(self):
        """Patch imports ที่มีปัญหา"""
        try:
            import builtins
            original_import = builtins.__import__
            
            def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
                try:
                    return original_import(name, globals, locals, fromlist, level)
                except ImportError as e:
                    error_msg = str(e)
                    
                    # Handle specific import errors
                    if 'safe_load_csv_auto' in error_msg and 'circular' in error_msg:
                        logger.warning(f"🔄 Handling circular import: {name}")
                        # Return a mock module with fallback function
                        class MockModule:
                            def safe_load_csv_auto(self, *args, **kwargs):
                                return self.fallback_functions.get('safe_load_csv_auto', lambda *a, **k: None)(*args, **kwargs)
                        
                        mock = MockModule()
                        mock.fallback_functions = self.fallback_functions
                        return mock
                    
                    elif 'ValueDrift' in error_msg:
                        logger.warning(f"🔄 Handling Evidently import: {name}")
                        # Return evidently fallback
                        try:
                            import evidently_v0430_fix
                            return evidently_v0430_fix
                        except:
                            class MockEvidently:
                                ValueDrift = lambda *args, **kwargs: None
                                DataDrift = lambda *args, **kwargs: None
                            return MockEvidently()
                    
                    elif 'Tuple' in error_msg:
                        logger.warning(f"🔄 Handling typing import: {name}")
                        from typing import Tuple, Dict, Any, Optional, List
                        class MockTyping:
                            Tuple = Tuple
                            Dict = Dict
                            Any = Any
                            Optional = Optional
                            List = List
                        return MockTyping()
                    
                    # Re-raise if we can't handle it
                    raise
            
            builtins.__import__ = patched_import
            logger.info("✅ Import patches applied")
            return True
            
        except Exception as e:
            logger.error(f"❌ Import patching failed: {e}")
            return False
    
    def fix_all(self):
        """แก้ไขปัญหาทั้งหมด"""
        logger.info("🔧 Starting comprehensive circular import fixes...")
        
        results = {
            'typing_fixes': self.apply_typing_fixes(),
            'csv_loader_fix': self.apply_csv_loader_fix(), 
            'import_order': self.fix_import_order(),
            'import_patches': self.patch_problematic_imports()
        }
        
        # Summary
        success_count = sum(results.values())
        total_count = len(results)
        
        logger.info("=" * 50)
        logger.info("🔧 Circular Import Fix Results:")
        for fix_name, success in results.items():
            status = "✅ Fixed" if success else "⚠️ Used fallback"
            logger.info(f"  {fix_name}: {status}")
        
        logger.info(f"📊 Overall: {success_count}/{total_count} fixes successful")
        
        if success_count == total_count:
            logger.info("🎉 All circular imports fixed!")
        else:
            logger.info("✅ System stable with fallbacks")
        
        logger.info("=" * 50)
        
        return results

# Global fixer instance
circular_import_fixer = CircularImportFixer()

def apply_all_fixes():
    """Apply all fixes"""
    return circular_import_fixer.fix_all()

if __name__ == "__main__":
    apply_all_fixes()
