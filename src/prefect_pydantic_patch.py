#!/usr/bin/env python3
        from pydantic import Field
import logging
        import pydantic
import sys
"""
Prefect Compatibility Patch for Pydantic v2
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

This module patches Pydantic to provide SecretField compatibility
for Prefect when using Pydantic v2.
"""


logger = logging.getLogger(__name__)


def patch_pydantic_for_prefect():
    """
    Monkey patch Pydantic to add SecretField compatibility for Prefect.
    This must be called before importing Prefect.
    """
    try:

        # Check if SecretField already exists
        if hasattr(pydantic, "SecretField"):
            logger.debug("Pydantic SecretField already available")
            return True

        # Create SecretField as an alias to Field for compatibility

        def SecretField(default = None, **kwargs):
            """Compatibility SecretField for Prefect with Pydantic v2"""
            # Remove any old v1 specific parameters
            kwargs.pop("secret", None)
            kwargs.pop("repr", None)
            return Field(default = default, **kwargs)

        # Inject SecretField into pydantic module
        pydantic.SecretField = SecretField

        # Also add to __all__ if it exists
        if hasattr(pydantic, "__all__"):
            if "SecretField" not in pydantic.__all__:
                pydantic.__all__.append("SecretField")

        logger.info(
            "Successfully patched Pydantic with SecretField for Prefect compatibility"
        )
        return True

    except ImportError:
        logger.warning("Pydantic not available for patching")
        return False
    except Exception as e:
        logger.error(f"Failed to patch Pydantic: {e}")
        return False


# Apply patch immediately when this module is imported
patch_pydantic_for_prefect()

# Alias for backward compatibility
monkey_patch_secretfield = patch_pydantic_for_prefect

# Export both function names
__all__ = ["patch_pydantic_for_prefect", "monkey_patch_secretfield"]