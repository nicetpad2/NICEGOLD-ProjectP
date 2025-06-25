        from pydantic import __version__ as pydantic_version
        from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField
            from pydantic import SecretStr
            from pydantic.fields import SecretField as PydanticSecretField
from typing import Any, Dict, Optional
            import json
import logging
"""
Pydantic Compatibility Layer - Support v1 and v2
"""


logger = logging.getLogger(__name__)

# Global compatibility variables
SecretField = None
BaseModel = None
Field = None

def initialize_pydantic():
    global SecretField, BaseModel, Field

    # Try Pydantic v2
    try:

        # In Pydantic v2, SecretField is replaced with SecretStr and Field
        try:

            def SecretFieldV2(default = None, **kwargs):
                """Pydantic v2 compatible SecretField"""
                return PydanticField(default = default, **kwargs)

            SecretField = SecretFieldV2
            BaseModel = PydanticBaseModel
            Field = PydanticField

            logger.info(f"✅ Pydantic v2 ({pydantic_version}) compatibility loaded")
            return True

        except ImportError:
            # Fall back to regular Field
            SecretField = PydanticField
            BaseModel = PydanticBaseModel
            Field = PydanticField

            logger.info(f"✅ Pydantic v2 ({pydantic_version}) with Field fallback")
            return True

    except ImportError:
        pass

    # Try Pydantic v1
    try:

        # Try to import SecretField from v1
        try:
            SecretField = PydanticSecretField
            logger.info("✅ Pydantic v1 SecretField found")
        except ImportError:
            # Use Field as fallback
            SecretField = PydanticField
            logger.info("✅ Pydantic v1 using Field as SecretField")

        BaseModel = PydanticBaseModel
        Field = PydanticField

        logger.info("✅ Pydantic v1 compatibility loaded")
        return True

    except ImportError:
        pass

    # Complete fallback
    logger.warning("⚠️ No Pydantic found, using complete fallback")

    def fallback_field(default = None, **kwargs):
        return default

    class FallbackBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

        def json(self):
            return json.dumps(self.dict())

    SecretField = fallback_field
    BaseModel = FallbackBaseModel
    Field = fallback_field

    return False

# Initialize on import
initialize_pydantic()

# Export
__all__ = ['SecretField', 'BaseModel', 'Field', 'initialize_pydantic']