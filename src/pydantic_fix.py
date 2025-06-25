            from pydantic import BaseModel as PydanticBaseModel
            from pydantic import Field as PydanticField
            from pydantic import Field as PydanticField, BaseModel as PydanticBaseModel
    from pydantic import SecretField, Field, BaseModel
        from src.pydantic_fix import SecretField, Field, BaseModel
from typing import Any, Optional
import logging
        import pydantic
import warnings
"""
Pydantic Compatibility Fix
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
แก้ไขปัญหา SecretField ใน Pydantic v2
"""


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
        version = pydantic.__version__
        logger.info(f"📦 Pydantic {version} detected")

        if version.startswith('2.'):
            # Pydantic v2

            def SecretFieldV2(default = None, **kwargs):
                """SecretField สำหรับ Pydantic v2"""
                # ลบ parameters ที่ไม่มีใน v2
                kwargs.pop('secret', None)
                kwargs.pop('repr', None)
                return PydanticField(default = default, **kwargs)

            SecretField = SecretFieldV2
            Field = PydanticField
            BaseModel = PydanticBaseModel

            logger.info("✅ Pydantic v2 compatibility ready")

        else:
            # Pydantic v1
            try:
except ImportError:
    try:
    except ImportError:
        # Fallback
        def SecretField(default = None, **kwargs): return default
        def Field(default = None, **kwargs): return default
        class BaseModel: pass as PydanticSecretField

            SecretField = PydanticSecretField
            Field = PydanticField
            BaseModel = PydanticBaseModel

            logger.info("✅ Pydantic v1 ready")

    except ImportError:
        # ไม่มี Pydantic - ใช้ fallback
        logger.warning("⚠️ No Pydantic found, using fallback")

        def FallbackSecretField(default = None, **kwargs):
            return default

        def FallbackField(default = None, **kwargs):
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