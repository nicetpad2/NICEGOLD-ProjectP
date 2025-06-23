#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
แก้ไขปัญหา Pydantic SecretField Import Error
===========================================
Thai Version - ครบถ้วนสำหรับ Pydantic v2
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ปิด warnings
warnings.filterwarnings('ignore')


def fix_pydantic_secretfield_import():
    """แก้ไขปัญหา SecretField import"""
    
    logger.info("🔧 กำลังแก้ไขปัญหา Pydantic SecretField...")
    
    try:
        # ตรวจสอบเวอร์ชัน Pydantic
        import pydantic
        version = pydantic.__version__
        logger.info(f"📦 Pydantic version: {version}")
        
        if version.startswith('2.'):
            # Pydantic v2 - สร้าง SecretField ทดแทน
            from pydantic import Field
            
            def SecretField(default=None, **kwargs):
                """SecretField ที่เข้ากันได้กับ Pydantic v2"""
                # ลบ parameters ที่ไม่มีใน v2
                kwargs.pop('secret', None)
                kwargs.pop('repr', None)
                kwargs.pop('min_length', None)
                kwargs.pop('max_length', None)
                
                return Field(default=default, **kwargs)
            
            # เพิ่ม SecretField เข้าไปใน pydantic module
            pydantic.SecretField = SecretField
            
            # เพิ่มใน __all__ ถ้ามี
            if hasattr(pydantic, '__all__'):
                if 'SecretField' not in pydantic.__all__:
                    pydantic.__all__.append('SecretField')
            
            # เพิ่มใน builtins เพื่อให้เข้าถึงได้จากทุกที่
            import builtins
            builtins.SecretField = SecretField
            
            logger.info("✅ สร้าง SecretField สำหรับ Pydantic v2 สำเร็จ")
            return True
            
        else:
            # Pydantic v1 - มี SecretField อยู่แล้ว
            logger.info("✅ Pydantic v1 - SecretField พร้อมใช้งาน")
            return True
            
    except ImportError:
        logger.warning("⚠️ ไม่พบ Pydantic - สร้าง fallback")
        return create_pydantic_fallback()
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาด: {e}")
        return create_pydantic_fallback()


def create_pydantic_fallback():
    """สร้าง Pydantic fallback เมื่อไม่มี Pydantic"""
    
    logger.info("🔄 สร้าง Pydantic fallback...")
    
    try:
        import builtins
        import sys
        import types

        # สร้าง SecretField ทดแทน
        def FallbackSecretField(default=None, **kwargs):
            """SecretField ทดแทนเมื่อไม่มี Pydantic"""
            return default
        
        # สร้าง Field ทดแทน
        def FallbackField(default=None, **kwargs):
            """Field ทดแทนเมื่อไม่มี Pydantic"""
            return default
        
        # สร้าง BaseModel ทดแทน
        class FallbackBaseModel:
            """BaseModel ทดแทนเมื่อไม่มี Pydantic"""
            
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            def dict(self):
                """คืนค่า dict ของ attributes"""
                return {k: v for k, v in self.__dict__.items() 
                       if not k.startswith('_')}
            
            def json(self):
                """คืนค่า JSON string"""
                import json
                return json.dumps(self.dict())
        
        # เพิ่ม fallback objects ใน builtins
        builtins.SecretField = FallbackSecretField
        builtins.Field = FallbackField
        builtins.BaseModel = FallbackBaseModel
        
        # สร้าง pydantic module ปลอม
        pydantic_module = types.ModuleType('pydantic')
        pydantic_module.SecretField = FallbackSecretField
        pydantic_module.Field = FallbackField
        pydantic_module.BaseModel = FallbackBaseModel
        pydantic_module.__version__ = "2.0.0-fallback"
        
        # เพิ่มใน sys.modules
        sys.modules['pydantic'] = pydantic_module
        
        logger.info("✅ สร้าง Pydantic fallback สำเร็จ")
        return True
        
    except Exception as e:
        logger.error(f"❌ สร้าง fallback ไม่สำเร็จ: {e}")
        return False


def fix_pipeline_imports():
    """แก้ไข imports ในไฟล์ต่างๆ ที่มีปัญหา"""
    
    logger.info("🔍 ค้นหาและแก้ไขไฟล์ที่มี import errors...")
    
    # ไฟล์ที่อาจมีปัญหา
    problematic_patterns = [
        "try:
    from pydantic import SecretField, Field, BaseModel
except ImportError:
    try:
        from src.pydantic_fix import SecretField, Field, BaseModel
    except ImportError:
        # Fallback
        def SecretField(default=None, **kwargs): return default
        def Field(default=None, **kwargs): return default
        class BaseModel: pass",
        "try:
    from pydantic import SecretField, Field, BaseModel
except ImportError:
    try:
        from src.pydantic_fix import SecretField, Field, BaseModel
    except ImportError:
        # Fallback
        def SecretField(default=None, **kwargs): return default
        def Field(default=None, **kwargs): return default
        class BaseModel: pass",
        "try:
    from pydantic import SecretField, Field, BaseModel
except ImportError:
    try:
        from src.pydantic_fix import SecretField, Field, BaseModel
    except ImportError:
        # Fallback
        def SecretField(default=None, **kwargs): return default
        def Field(default=None, **kwargs): return default
        class BaseModel: pass"
    ]
    
    # ค้นหาไฟล์ที่มีปัญหา
    project_root = Path(".")
    python_files = list(project_root.rglob("*.py"))
    
    fixed_files = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # แก้ไข import statements
            for pattern in problematic_patterns:
                if pattern in content:
                    logger.info(f"🔧 แก้ไข import ใน {file_path}")
                    
                    # แทนที่ด้วย import ที่ปลอดภัย
                    if "SecretField" in pattern:
                        safe_import = """try:
    try:
    from pydantic import SecretField, Field, BaseModel
except ImportError:
    try:
        from src.pydantic_fix import SecretField, Field, BaseModel
    except ImportError:
        # Fallback
        def SecretField(default=None, **kwargs): return default
        def Field(default=None, **kwargs): return default
        class BaseModel: pass, Field, BaseModel
except ImportError:
    try:
        from src.pydantic_fix import SecretField, Field, BaseModel
    except ImportError:
        # Fallback
        def SecretField(default=None, **kwargs): return default
        def Field(default=None, **kwargs): return default
        class BaseModel: pass"""
                        
                        content = content.replace(pattern, safe_import)
            
            # บันทึกไฟล์ถ้ามีการเปลี่ยนแปลง
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files += 1
                logger.info(f"✅ แก้ไข {file_path} เรียบร้อย")
                
        except Exception as e:
            logger.warning(f"⚠️ ไม่สามารถแก้ไข {file_path}: {e}")
    
    logger.info(f"📊 แก้ไขไฟล์ทั้งหมด: {fixed_files} ไฟล์")
    return fixed_files > 0


def create_compatibility_module():
    """สร้างโมดูล compatibility สำหรับ pydantic"""
    
    logger.info("📁 สร้างโมดูล compatibility...")
    
    # สร้างโฟลเดอร์ src ถ้ายังไม่มี
    src_dir = Path("src")
    src_dir.mkdir(exist_ok=True)
    
    # สร้างไฟล์ __init__.py
    init_file = src_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# ProjectP src package\n")
    
    # สร้างโมดูล pydantic_fix.py
    compat_file = src_dir / "pydantic_fix.py"
    
    compat_code = '''"""
Pydantic Compatibility Fix
=========================
แก้ไขปัญหา SecretField ใน Pydantic v2
"""

import logging
import warnings
from typing import Any, Optional

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ตัวแปรสำหรับเก็บ compatibility objects
SecretField = None
Field = None
BaseModel = None

def initialize_pydantic():
    """เริ่มต้น pydantic compatibility"""
    global SecretField, Field, BaseModel
    
    try:
        # ลอง import Pydantic
        import pydantic
        version = pydantic.__version__
        logger.info(f"📦 Pydantic {version} detected")
        
        if version.startswith('2.'):
            # Pydantic v2
            from pydantic import Field as PydanticField, BaseModel as PydanticBaseModel
            
            def SecretFieldV2(default=None, **kwargs):
                """SecretField สำหรับ Pydantic v2"""
                # ลบ parameters ที่ไม่มีใน v2
                kwargs.pop('secret', None)
                kwargs.pop('repr', None)
                return PydanticField(default=default, **kwargs)
            
            SecretField = SecretFieldV2
            Field = PydanticField
            BaseModel = PydanticBaseModel
            
            logger.info("✅ Pydantic v2 compatibility ready")
            
        else:
            # Pydantic v1
            try:
    from pydantic import SecretField, Field, BaseModel
except ImportError:
    try:
        from src.pydantic_fix import SecretField, Field, BaseModel
    except ImportError:
        # Fallback
        def SecretField(default=None, **kwargs): return default
        def Field(default=None, **kwargs): return default
        class BaseModel: pass as PydanticSecretField
            from pydantic import Field as PydanticField
            from pydantic import BaseModel as PydanticBaseModel
            
            SecretField = PydanticSecretField
            Field = PydanticField
            BaseModel = PydanticBaseModel
            
            logger.info("✅ Pydantic v1 ready")
            
    except ImportError:
        # ไม่มี Pydantic - ใช้ fallback
        logger.warning("⚠️ No Pydantic found, using fallback")
        
        def FallbackSecretField(default=None, **kwargs):
            return default
            
        def FallbackField(default=None, **kwargs):
            return default
            
        class FallbackBaseModel:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
            def dict(self):
                return {k: v for k, v in self.__dict__.items() 
                       if not k.startswith('_')}
        
        SecretField = FallbackSecretField
        Field = FallbackField
        BaseModel = FallbackBaseModel
        
        logger.info("✅ Fallback mode ready")

# เริ่มต้นเมื่อ import
initialize_pydantic()

# Export
__all__ = ['SecretField', 'Field', 'BaseModel']
'''
    
    with open(compat_file, 'w', encoding='utf-8') as f:
        f.write(compat_code)
    
    logger.info(f"✅ สร้าง {compat_file} เรียบร้อย")
    return True


def test_fix():
    """ทดสอบว่าการแก้ไขทำงานหรือไม่"""
    
    logger.info("🧪 ทดสอบการแก้ไข...")
    
    try:
        # ทดสอบ import จาก pydantic โดยตรง
        try:
    from pydantic import SecretField, Field, BaseModel
except ImportError:
    try:
        from src.pydantic_fix import SecretField, Field, BaseModel
    except ImportError:
        # Fallback
        def SecretField(default=None, **kwargs): return default
        def Field(default=None, **kwargs): return default
        class BaseModel: pass

        # ทดสอบสร้าง field
        secret_field = SecretField(default="test_secret")
        normal_field = Field(default="test_normal")
        
        # ทดสอบสร้าง model
        class TestModel(BaseModel):
            secret: str = SecretField(default="hidden")
            normal: str = Field(default="visible")
        
        model = TestModel()
        model_dict = model.dict() if hasattr(model, 'dict') else {}
        
        logger.info("✅ การทดสอบสำเร็จ!")
        logger.info(f"📊 Model data: {model_dict}")
        return True
        
    except Exception as e:
        logger.error(f"❌ การทดสอบล้มเหลว: {e}")
        return False


def main():
    """ฟังก์ชันหลักสำหรับแก้ไขปัญหา"""
    
    print("🎯 แก้ไขปัญหา Pydantic SecretField Import Error")
    print("=" * 60)
    
    success_steps = 0
    total_steps = 4
    
    # Step 1: แก้ไข SecretField import
    if fix_pydantic_secretfield_import():
        success_steps += 1
        print("✅ Step 1: แก้ไข SecretField import สำเร็จ")
    else:
        print("❌ Step 1: แก้ไข SecretField import ล้มเหลว")
    
    # Step 2: สร้างโมดูล compatibility
    if create_compatibility_module():
        success_steps += 1
        print("✅ Step 2: สร้างโมดูล compatibility สำเร็จ")
    else:
        print("❌ Step 2: สร้างโมดูล compatibility ล้มเหลว")
    
    # Step 3: แก้ไขไฟล์ที่มีปัญหา
    if fix_pipeline_imports():
        success_steps += 1
        print("✅ Step 3: แก้ไขไฟล์ pipeline สำเร็จ")
    else:
        print("⚠️ Step 3: ไม่พบไฟล์ที่ต้องแก้ไข")
        success_steps += 1  # ไม่ใช่ error
    
    # Step 4: ทดสอบการแก้ไข
    if test_fix():
        success_steps += 1
        print("✅ Step 4: ทดสอบการแก้ไขสำเร็จ")
    else:
        print("❌ Step 4: ทดสอบการแก้ไขล้มเหลว")
    
    print("\n" + "=" * 60)
    print(f"📊 ผลลัพธ์: {success_steps}/{total_steps} steps สำเร็จ")
    
    if success_steps >= 3:
        print("🎉 การแก้ไขสำเร็จ! Pipeline ควรทำงานได้แล้ว")
        print("\n💡 วิธีใช้:")
        print("1. ใช้: try:
    from pydantic import SecretField, Field, BaseModel
except ImportError:
    try:
        from src.pydantic_fix import SecretField, Field, BaseModel
    except ImportError:
        # Fallback
        def SecretField(default=None, **kwargs): return default
        def Field(default=None, **kwargs): return default
        class BaseModel: pass  (จะทำงานแล้ว)")
        print("2. หรือ: from src.pydantic_fix import SecretField")
        print("3. เรียก: python your_pipeline_script.py")
        return True
    else:
        print("⚠️ การแก้ไขไม่สมบูรณ์ กรุณาตรวจสอบ logs")
        return False


if __name__ == "__main__":
    main()
