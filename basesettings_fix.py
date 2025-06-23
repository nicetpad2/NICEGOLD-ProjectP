#!/usr/bin/env python3
"""
Pydantic BaseSettings Compatibility Fix
======================================
แก้ไขปัญหา BaseSettings ที่ถูกย้ายไปยัง pydantic-settings ใน Pydantic v2
พร้อมสำหรับ Production
"""

import logging
import sys
import warnings
from typing import Any, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


def create_basesettings_compatibility():
    """Create BaseSettings compatibility for Pydantic v2"""

    # Strategy 1: Try pydantic-settings package (recommended for v2)
    try:
        from pydantic_settings import BaseSettings

        logger.info("✅ Using BaseSettings from pydantic-settings (recommended)")
        return BaseSettings
    except ImportError:
        logger.info("⚠️ pydantic-settings not available, trying alternatives...")

    # Strategy 2: Try old Pydantic v1 location
    try:
        from pydantic import BaseSettings

        logger.info("✅ Using BaseSettings from pydantic (v1 compatibility)")
        return BaseSettings
    except ImportError:
        logger.info("⚠️ BaseSettings not available in pydantic, creating fallback...")

    # Strategy 3: Create fallback BaseSettings
    try:
        from pydantic import BaseModel

        class BaseSettingsFallback(BaseModel):
            """Fallback BaseSettings implementation"""

            class Config:
                env_file = ".env"
                env_file_encoding = "utf-8"
                case_sensitive = False

            @classmethod
            def from_env(cls, **kwargs):
                """Load settings from environment variables"""
                import os

                env_values = {}

                # Get field names from model
                if hasattr(cls, "__fields__"):
                    # Pydantic v1 style
                    field_names = cls.__fields__.keys()
                elif hasattr(cls, "model_fields"):
                    # Pydantic v2 style
                    field_names = cls.model_fields.keys()
                else:
                    field_names = []

                # Load from environment
                for field_name in field_names:
                    env_name = field_name.upper()
                    if env_name in os.environ:
                        env_values[field_name] = os.environ[env_name]

                # Merge with provided kwargs
                env_values.update(kwargs)
                return cls(**env_values)

        logger.info("✅ Created BaseSettings fallback implementation")
        return BaseSettingsFallback

    except ImportError:
        logger.warning(
            "⚠️ Cannot create BaseSettings fallback, using minimal implementation"
        )

        # Strategy 4: Minimal fallback
        class MinimalBaseSettings:
            """Minimal BaseSettings implementation"""

            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            @classmethod
            def from_env(cls, **kwargs):
                import os

                env_values = {}

                # Load common environment variables
                common_env_vars = [
                    "DATABASE_URL",
                    "API_KEY",
                    "SECRET_KEY",
                    "DEBUG",
                    "HOST",
                    "PORT",
                    "LOG_LEVEL",
                ]

                for var in common_env_vars:
                    if var in os.environ:
                        env_values[var.lower()] = os.environ[var]

                env_values.update(kwargs)
                return cls(**env_values)

        logger.info("✅ Created minimal BaseSettings implementation")
        return MinimalBaseSettings


def install_pydantic_settings():
    """Try to install pydantic-settings package"""
    try:
        import subprocess
        import sys

        logger.info("🔧 Attempting to install pydantic-settings...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pydantic-settings"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info("✅ pydantic-settings installed successfully")
            return True
        else:
            logger.warning(f"⚠️ Failed to install pydantic-settings: {result.stderr}")
            return False

    except Exception as e:
        logger.warning(f"⚠️ Could not install pydantic-settings: {e}")
        return False


def update_compatibility_layer():
    """Update the main Pydantic compatibility layer with BaseSettings"""

    try:
        # Read the current compatibility file
        compat_file = "src/pydantic_v2_compat.py"

        with open(compat_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if BaseSettings is already handled
        if "BaseSettings" in content:
            logger.info("✅ BaseSettings already handled in compatibility layer")
            return True

        # Add BaseSettings to the compatibility layer
        basesettings_code = '''
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
'''

        # Insert before the final exports
        if "__all__" in content:
            # Insert before __all__
            insertion_point = content.rfind("__all__")
            new_content = (
                content[:insertion_point]
                + basesettings_code
                + "\n\n"
                + content[insertion_point:]
            )

            # Update __all__ to include BaseSettings
            new_content = new_content.replace(
                '__all__ = ["SecretField", "Field", "BaseModel"]',
                '__all__ = ["SecretField", "Field", "BaseModel", "BaseSettings"]',
            )
        else:
            # Append at the end
            new_content = (
                content
                + "\n\n"
                + basesettings_code
                + '\n\n__all__ = ["SecretField", "Field", "BaseModel", "BaseSettings"]\n'
            )

        # Write back
        with open(compat_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.info("✅ Updated compatibility layer with BaseSettings")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to update compatibility layer: {e}")
        return False


def test_basesettings_compatibility():
    """Test BaseSettings compatibility"""

    try:
        # Test the compatibility layer
        BaseSettings = create_basesettings_compatibility()

        # Create a test settings class
        class TestSettings(BaseSettings):
            app_name: str = "TestApp"
            debug: bool = False
            api_key: str = "default-key"

        # Test instantiation
        settings = TestSettings()
        logger.info(f"✅ BaseSettings test passed: {settings.app_name}")

        # Test from_env if available
        if hasattr(TestSettings, "from_env"):
            env_settings = TestSettings.from_env(app_name="EnvApp")
            logger.info(
                f"✅ BaseSettings from_env test passed: {env_settings.app_name}"
            )

        return True

    except Exception as e:
        logger.error(f"❌ BaseSettings test failed: {e}")
        return False


def main():
    """Main BaseSettings compatibility fix"""

    logger.info("🚀 Pydantic BaseSettings Compatibility Fix")
    logger.info("=" * 60)

    # Step 1: Try to install pydantic-settings
    installed = install_pydantic_settings()

    # Step 2: Create compatibility layer
    BaseSettings = create_basesettings_compatibility()

    # Step 3: Update main compatibility file
    updated = update_compatibility_layer()

    # Step 4: Test compatibility
    test_passed = test_basesettings_compatibility()

    # Report results
    logger.info("=" * 60)
    logger.info("📊 BASESETTINGS COMPATIBILITY RESULTS")
    logger.info("=" * 60)

    results = {
        "pydantic-settings_install": installed,
        "compatibility_layer": BaseSettings is not None,
        "main_file_update": updated,
        "compatibility_test": test_passed,
    }

    all_ok = True
    for component, status in results.items():
        icon = "✅ OK" if status else "❌ FAILED"
        logger.info(f"{component:25} : {icon}")
        if not status:
            all_ok = False

    logger.info("=" * 60)

    if all_ok:
        logger.info("🎉 BASESETTINGS COMPATIBILITY READY!")
        logger.info("✅ BaseSettings is now available for use")
        print("\n🎉 BaseSettings compatibility fix completed!")
        print("✅ You can now use BaseSettings in your code")
        print("💡 Import: from src.pydantic_v2_compat import BaseSettings")
        return True
    else:
        logger.error("❌ Some BaseSettings compatibility issues remain")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
