"""
Pydantic Compatibility Fix for SecretField
แก้ไขปัญหา SecretField ในทุกเวอร์ชันของ pydantic
"""

import logging
import warnings
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Global variables for pydantic compatibility
PYDANTIC_V1 = False
PYDANTIC_V2 = False
FALLBACK_MODE = False

def detect_pydantic_version():
    """ตรวจสอบเวอร์ชันของ pydantic"""
    global PYDANTIC_V1, PYDANTIC_V2, FALLBACK_MODE
    
    try:
        import pydantic
        version = pydantic.VERSION
        
        if version.startswith('1.'):
            PYDANTIC_V1 = True
            logger.info(f"✅ Pydantic v1 detected: {version}")
        elif version.startswith('2.'):
            PYDANTIC_V2 = True
            logger.info(f"✅ Pydantic v2 detected: {version}")
        else:
            logger.warning(f"⚠️ Unknown pydantic version: {version}")
            FALLBACK_MODE = True
    except ImportError:
        logger.warning("⚠️ Pydantic not available")
        FALLBACK_MODE = True

def get_secret_field():
    """ได้ SecretField ที่เข้ากันได้กับทุกเวอร์ชัน"""
    global PYDANTIC_V1, PYDANTIC_V2, FALLBACK_MODE
    
    if PYDANTIC_V1:
        try:
            from pydantic import SecretStr
            from pydantic.fields import Field
            
            def SecretField(default=None, **kwargs):
                return Field(default=default, **kwargs)
            
            logger.info("✅ Using Pydantic v1 compatibility")
            return SecretField, Field, SecretStr
            
        except ImportError:
            try:
                from pydantic import Field
                def SecretField(default=None, **kwargs):
                    return Field(default=default, **kwargs)
                logger.info("✅ Using Pydantic v1 Field only")
                return SecretField, Field, str
            except ImportError:
                pass
    
    elif PYDANTIC_V2:
        try:
            from pydantic import Field, SecretStr
            
            def SecretField(default=None, **kwargs):
                return Field(default=default, **kwargs)
            
            logger.info("✅ Using Pydantic v2 compatibility")
            return SecretField, Field, SecretStr
            
        except ImportError:
            pass
    
    # Fallback mode
    logger.warning("⚠️ Using pydantic fallback mode")
    
    def FallbackSecretField(default=None, **kwargs):
        return default
    
    def FallbackField(default=None, **kwargs):
        return default
    
    class FallbackSecretStr:
        def __init__(self, value):
            self._value = str(value)
        
        def get_secret_value(self):
            return self._value
        
        def __str__(self):
            return '***'
    
    return FallbackSecretField, FallbackField, FallbackSecretStr

def get_base_model():
    """ได้ BaseModel ที่เข้ากันได้"""
    try:
        from pydantic import BaseModel
        return BaseModel
    except ImportError:
        logger.warning("⚠️ Using BaseModel fallback")
        
        class FallbackBaseModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            def dict(self, **kwargs):
                return {k: v for k, v in self.__dict__.items() 
                       if not k.startswith('_')}
            
            def json(self, **kwargs):
                import json
                return json.dumps(self.dict())
            
            class Config:
                arbitrary_types_allowed = True
                extra = 'allow'
        
        return FallbackBaseModel

# Initialize
detect_pydantic_version()
SecretField, Field, SecretStr = get_secret_field()
BaseModel = get_base_model()

# Make them available globally
import builtins
builtins.SecretField = SecretField
builtins.PydanticField = Field
builtins.PydanticSecretStr = SecretStr
builtins.PydanticBaseModel = BaseModel

logger.info("✅ Pydantic compatibility layer ready")
