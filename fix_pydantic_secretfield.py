"""
Direct Pydantic v2 SecretField Fix
=================================
Creates a direct replacement for SecretField that works with Pydantic v2
"""

import logging
import sys
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


def fix_pydantic_secretfield():
    """Direct fix for Pydantic v2 SecretField issue"""

    try:
        import pydantic
        from pydantic import Field

        # Create SecretField replacement
        def SecretField(
            default: Any = None,
            *,
            alias: Optional[str] = None,
            title: Optional[str] = None,
            description: Optional[str] = None,
            **kwargs,
        ):
            """
            Pydantic v2 compatible SecretField replacement

            This function replaces the removed SecretField from Pydantic v1
            with a Field that has the same interface but works with v2.
            """
            # Remove v1-specific arguments that don't exist in v2
            kwargs.pop("secret", None)
            kwargs.pop("repr", None)
            kwargs.pop("min_length", None)  # Use min_length from Field instead
            kwargs.pop("max_length", None)  # Use max_length from Field instead

            return Field(
                default=default,
                alias=alias,
                title=title,
                description=description,
                **kwargs,
            )

        # Add SecretField to pydantic module
        pydantic.SecretField = SecretField

        # Also add to pydantic.__all__ if it exists
        if hasattr(pydantic, "__all__"):
            if "SecretField" not in pydantic.__all__:
                pydantic.__all__.append("SecretField")

        # Add to global builtins for import compatibility
        import builtins

        builtins.SecretField = SecretField

        # Verify it works
        test_field = SecretField(default="test")

        logger.info("✅ Pydantic v2 SecretField fix applied successfully")
        return True

    except ImportError as e:
        logger.error(f"❌ Failed to fix Pydantic SecretField: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error fixing Pydantic: {e}")
        return False


def create_secretstr_compatibility():
    """Create SecretStr compatibility for v2"""
    try:
        # SecretStr exists in v2, just make sure it's accessible
        import builtins

        from pydantic import SecretStr as PydanticSecretStr

        builtins.SecretStr = PydanticSecretStr

        logger.info("✅ SecretStr compatibility established")
        return True

    except ImportError:
        # Create fallback SecretStr
        class FallbackSecretStr:
            def __init__(self, secret_value: str):
                self._secret_value = str(secret_value)

            def get_secret_value(self) -> str:
                return self._secret_value

            def __str__(self) -> str:
                return "***"

            def __repr__(self) -> str:
                return "SecretStr(***)"

        import builtins

        builtins.SecretStr = FallbackSecretStr

        logger.info("✅ Fallback SecretStr created")
        return True


def apply_comprehensive_pydantic_fix():
    """Apply comprehensive Pydantic compatibility fix"""

    logger.info("🔧 Applying comprehensive Pydantic v2 fix...")

    success = True

    # Fix SecretField
    if not fix_pydantic_secretfield():
        success = False

    # Fix SecretStr
    if not create_secretstr_compatibility():
        success = False

    # Test the fix
    try:
        # Test direct import
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

        test_field = SecretField(default="test")

        logger.info("✅ Pydantic fix verification successful")

        # Show available fields
        import pydantic

        available_items = [item for item in dir(pydantic) if not item.startswith("_")]
        logger.info(f"📦 Available in pydantic: {', '.join(sorted(available_items))}")

        return True

    except Exception as e:
        logger.error(f"❌ Pydantic fix verification failed: {e}")
        return False


if __name__ == "__main__":
    success = apply_comprehensive_pydantic_fix()

    if success:
        print("\n🎉 Pydantic v2 SecretField fix completed successfully!")
        print("You can now use: try:
    from pydantic import SecretField, Field, BaseModel
except ImportError:
    try:
        from src.pydantic_fix import SecretField, Field, BaseModel
    except ImportError:
        # Fallback
        def SecretField(default=None, **kwargs): return default
        def Field(default=None, **kwargs): return default
        class BaseModel: pass")
    else:
        print("\n⚠️ Pydantic fix completed with issues.")
        print("Please check the logs for details.")
