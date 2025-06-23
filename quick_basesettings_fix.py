#!/usr/bin/env python3
"""
Quick BaseSettings Compatibility Fix
===================================
แก้ไขปัญหา BaseSettings อย่างรวดเร็วสำหรับ Production
"""

import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_basesettings_fix():
    """Apply BaseSettings compatibility fix directly"""

    logger.info("🔧 Applying BaseSettings compatibility fix...")

    # Read the current compatibility file
    compat_file = "src/pydantic_v2_compat.py"

    try:
        with open(compat_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if BaseSettings is already handled
        if "BaseSettings" in content:
            logger.info("✅ BaseSettings already in compatibility layer")
            return True

        # Add BaseSettings import and handling
        basesettings_addition = """
# BaseSettings compatibility (moved to pydantic-settings in v2)
try:
    # Try pydantic-settings first (recommended for v2)
    from pydantic_settings import BaseSettings
    logger.info("✅ Using BaseSettings from pydantic-settings")
except ImportError:
    try:
        # Fallback to pydantic v1 location
        from pydantic import BaseSettings
        logger.info("✅ Using BaseSettings from pydantic (v1)")
    except ImportError:
        # Create minimal BaseSettings fallback
        class BaseSettings(BaseModel):
            '''Minimal BaseSettings implementation'''
            
            class Config:
                env_file = '.env'
                env_file_encoding = 'utf-8'
                case_sensitive = False
            
            @classmethod
            def from_env(cls, **kwargs):
                '''Load settings from environment variables'''
                import os
                env_values = {}
                
                # Load common environment variables
                env_vars = ['DEBUG', 'HOST', 'PORT', 'DATABASE_URL', 'API_KEY']
                for var in env_vars:
                    if var in os.environ:
                        env_values[var.lower()] = os.environ[var]
                
                env_values.update(kwargs)
                return cls(**env_values)
        
        logger.info("✅ Using BaseSettings fallback implementation")
"""

        # Find insertion point (before __all__ if it exists)
        if "__all__" in content:
            insertion_point = content.rfind("__all__")
            new_content = (
                content[:insertion_point]
                + basesettings_addition
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
                + "\n"
                + basesettings_addition
                + '\n\n__all__ = ["SecretField", "Field", "BaseModel", "BaseSettings"]\n'
            )

        # Write back
        with open(compat_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.info("✅ BaseSettings added to compatibility layer")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to update compatibility layer: {e}")
        return False


def test_basesettings_import():
    """Test BaseSettings import from compatibility layer"""

    try:
        from src.pydantic_v2_compat import BaseSettings

        logger.info("✅ BaseSettings import from compatibility layer successful")

        # Test creating a settings class
        class TestSettings(BaseSettings):
            app_name: str = "TestApp"
            debug: bool = False

        settings = TestSettings()
        logger.info(f"✅ BaseSettings instantiation successful: {settings.app_name}")
        return True

    except Exception as e:
        logger.error(f"❌ BaseSettings test failed: {e}")
        return False


def main():
    """Main quick BaseSettings fix"""

    logger.info("🚀 Quick BaseSettings Compatibility Fix")
    logger.info("=" * 50)

    # Apply the fix
    fix_applied = apply_basesettings_fix()

    if fix_applied:
        # Test the fix
        test_passed = test_basesettings_import()

        if test_passed:
            logger.info("=" * 50)
            logger.info("🎉 BASESETTINGS FIX SUCCESSFUL!")
            logger.info("✅ BaseSettings is now available")
            logger.info("💡 Import: from src.pydantic_v2_compat import BaseSettings")
            print("\n🎉 BaseSettings compatibility fix completed!")
            print("✅ Ready for production use")
            return True

    logger.error("❌ BaseSettings fix failed")
    return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
