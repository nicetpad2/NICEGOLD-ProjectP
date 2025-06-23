"""
Professional Pydantic v2 Compatibility Layer
==========================================
Handles the SecretField import issue in Pydantic v2.x where SecretField was removed
"""

import logging
import sys
import warnings
from typing import Any, Dict, Optional, Type, Union

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


class PydanticV2Compatibility:
    """Professional compatibility layer for Pydantic v2"""

    def __init__(self):
        self.pydantic_version = None
        self.compatibility_mode = None
        self._initialize()

    def _initialize(self):
        """Initialize compatibility layer with proper detection"""
        try:
            import pydantic

            self.pydantic_version = pydantic.__version__

            if self.pydantic_version.startswith("2."):
                self.compatibility_mode = "v2"
                self._setup_v2_compatibility()
            elif self.pydantic_version.startswith("1."):
                self.compatibility_mode = "v1"
                self._setup_v1_compatibility()
            else:
                self.compatibility_mode = "fallback"
                self._setup_fallback()

            logger.info(
                f"✅ Pydantic {self.pydantic_version} compatibility initialized ({self.compatibility_mode})"
            )

        except ImportError:
            self.compatibility_mode = "fallback"
            self._setup_fallback()
            logger.warning("⚠️ Pydantic not found, using fallback compatibility")

    def _setup_v2_compatibility(self):
        """Setup Pydantic v2 compatibility"""
        try:
            from pydantic import BaseModel, ConfigDict, Field
            from pydantic.types import SecretStr

            # Create SecretField replacement for v2
            def SecretField(default=None, **kwargs):
                """
                Pydantic v2 compatible SecretField replacement
                Maps to Field with appropriate configuration
                """
                # Remove v1-specific arguments that don't exist in v2
                kwargs.pop("secret", None)
                kwargs.pop("repr", None)

                return Field(default=default, **kwargs)

            # Create enhanced SecretStr wrapper
            class EnhancedSecretStr(SecretStr):
                """Enhanced SecretStr with v1 compatibility methods"""

                def __init__(self, secret_value: str):
                    super().__init__(secret_value)

                @classmethod
                def __get_validators__(cls):
                    """V1 compatibility method"""
                    yield cls.validate

                @classmethod
                def validate(cls, value):
                    """V1 style validation"""
                    if isinstance(value, cls):
                        return value
                    return cls(str(value))

            # Register in global namespace
            self._register_compatibility_objects(
                {
                    "SecretField": SecretField,
                    "Field": Field,
                    "BaseModel": BaseModel,
                    "SecretStr": EnhancedSecretStr,
                    "ConfigDict": ConfigDict,
                }
            )

        except ImportError as e:
            logger.error(f"❌ Failed to setup Pydantic v2 compatibility: {e}")
            self._setup_fallback()

    def _setup_v1_compatibility(self):
        """Setup Pydantic v1 compatibility"""
        try:
            from pydantic import BaseModel, Field

            # Try to import SecretField from v1
            try:
                from pydantic import SecretField

                logger.info("✅ Native SecretField found in v1")
            except ImportError:
                try:
                    from pydantic.fields import SecretField

                    logger.info("✅ SecretField imported from pydantic.fields")
                except ImportError:
                    # Create SecretField fallback for v1
                    def SecretField(default=None, **kwargs):
                        """V1 SecretField fallback using Field"""
                        return Field(default=default, **kwargs)

                    logger.info("✅ Created SecretField fallback for v1")

            # Import SecretStr
            try:
                from pydantic import SecretStr
            except ImportError:
                # Fallback SecretStr
                class SecretStr:
                    def __init__(self, value):
                        self._secret_value = str(value)

                    def get_secret_value(self):
                        return self._secret_value

                    def __str__(self):
                        return "***"

                    def __repr__(self):
                        return f"SecretStr(***)"

            # Register compatibility objects
            self._register_compatibility_objects(
                {
                    "SecretField": SecretField,
                    "Field": Field,
                    "BaseModel": BaseModel,
                    "SecretStr": SecretStr,
                }
            )

        except ImportError as e:
            logger.error(f"❌ Failed to setup Pydantic v1 compatibility: {e}")
            self._setup_fallback()

    def _setup_fallback(self):
        """Setup complete fallback when Pydantic is not available"""
        logger.info("🔄 Setting up Pydantic fallback compatibility")

        def SecretField(default=None, **kwargs):
            """Complete fallback SecretField"""
            return default

        def Field(default=None, **kwargs):
            """Complete fallback Field"""
            return default

        class FallbackSecretStr:
            """Fallback SecretStr implementation"""

            def __init__(self, value):
                self._secret_value = str(value) if value is not None else ""

            def get_secret_value(self):
                return self._secret_value

            def __str__(self):
                return "***"

            def __repr__(self):
                return "SecretStr(***)"

        class FallbackBaseModel:
            """Fallback BaseModel implementation"""

            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def dict(self, **kwargs):
                """Return dict representation"""
                return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

            def json(self, **kwargs):
                """Return JSON representation"""
                import json

                return json.dumps(self.dict())

            class Config:
                """Fallback config class"""

                arbitrary_types_allowed = True
                extra = "allow"

        # Register fallback objects
        self._register_compatibility_objects(
            {
                "SecretField": SecretField,
                "Field": Field,
                "BaseModel": FallbackBaseModel,
                "SecretStr": FallbackSecretStr,
            }
        )

    def _register_compatibility_objects(self, objects: Dict[str, Any]):
        """Register compatibility objects in multiple namespaces"""
        # Register in builtins for global access
        import builtins

        for name, obj in objects.items():
            setattr(builtins, f"Pydantic{name}", obj)
            # Also register without prefix for convenience
            if not hasattr(builtins, name):
                setattr(builtins, name, obj)

        # Monkey-patch pydantic module if it exists
        try:
            import pydantic

            for name, obj in objects.items():
                if not hasattr(pydantic, name):
                    setattr(pydantic, name, obj)
        except ImportError:
            # Create minimal pydantic module
            import types

            pydantic_module = types.ModuleType("pydantic")
            for name, obj in objects.items():
                setattr(pydantic_module, name, obj)
            sys.modules["pydantic"] = pydantic_module

        logger.info(f"✅ Registered {len(objects)} compatibility objects")

    def test_compatibility(self):
        """Test that compatibility objects work correctly"""
        try:
            from pydantic import BaseModel, Field, SecretField

            # Test SecretField
            secret_field = SecretField(default="test")

            # Test Field
            regular_field = Field(default="test")

            # Test BaseModel
            class TestModel(BaseModel):
                secret: str = SecretField(default="secret")
                regular: str = Field(default="regular")

            model = TestModel()
            logger.info("✅ Compatibility test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Compatibility test failed: {e}")
            return False

    def get_info(self):
        """Get compatibility information"""
        return {
            "pydantic_version": self.pydantic_version,
            "compatibility_mode": self.compatibility_mode,
            "available_objects": ["SecretField", "Field", "BaseModel", "SecretStr"],
        }


# Initialize compatibility layer
_compat = PydanticV2Compatibility()

# Export compatibility objects safely
try:
    from pydantic import BaseModel, Field

    # Don't try to import SecretField directly - it doesn't exist in v2
    # Instead, use our compatibility layer
    if hasattr(__import__("builtins"), "PydanticSecretField"):
        SecretField = getattr(__import__("builtins"), "PydanticSecretField")
    else:
        # Create fallback SecretField
        def SecretField(default=None, **kwargs):
            return Field(default=default, **kwargs)

    if hasattr(__import__("pydantic"), "SecretStr"):
        from pydantic import SecretStr
    else:
        from builtins import SecretStr

except ImportError:
    from builtins import BaseModel, Field, SecretField, SecretStr

# Test compatibility on import
if _compat.test_compatibility():
    logger.info("🎉 Pydantic compatibility successfully established")
else:
    logger.warning("⚠️ Pydantic compatibility established with limitations")


# BaseSettings compatibility (moved to pydantic-settings in v2)
BaseSettings = None

def _get_base_settings():
    """Get BaseSettings with compatibility handling"""
    global BaseSettings
    
    if BaseSettings is not None:
        return BaseSettings
    
    # Strategy 1: pydantic-settings (recommended for v2)
    try:
        from pydantic_settings import BaseSettings as PydanticSettings
        BaseSettings = PydanticSettings
        logger.info("✅ Using BaseSettings from pydantic-settings")
        return BaseSettings
    except ImportError:
        pass
    
    # Strategy 2: pydantic v1 location
    try:
        from pydantic import BaseSettings as PydanticBaseSettings
        BaseSettings = PydanticBaseSettings
        logger.info("✅ Using BaseSettings from pydantic (v1)")
        return BaseSettings
    except ImportError:
        pass
    
    # Strategy 3: Create fallback
    class BaseSettingsFallback(BaseModel):
        """Fallback BaseSettings implementation"""
        
        class Config:
            env_file = '.env'
            env_file_encoding = 'utf-8'
            case_sensitive = False
    
    BaseSettings = BaseSettingsFallback
    logger.info("✅ Using BaseSettings fallback")
    return BaseSettings

# Make BaseSettings available
try:
    BaseSettings = _get_base_settings()
except Exception as e:
    logger.warning(f"⚠️ BaseSettings setup failed: {e}")
    BaseSettings = BaseModel  # Ultimate fallback


__all__ = ["SecretField", "Field", "BaseModel", "SecretStr", "PydanticV2Compatibility"]
