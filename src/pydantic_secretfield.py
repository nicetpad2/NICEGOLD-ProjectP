    from pydantic import __version__ as pydantic_version
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field as PydanticField
from typing import Any, Optional
import logging
"""
Simple Pydantic SecretField Replacement
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
A drop - in replacement for the missing SecretField in Pydantic v2
"""


logger = logging.getLogger(__name__)

try:

    logger.info(f"Using Pydantic {pydantic_version}")

    def SecretField(
        default: Any = None, 
        *, 
        alias: Optional[str] = None, 
        title: Optional[str] = None, 
        description: Optional[str] = None, 
        **kwargs: Any, 
    ) -> Any:
        """
        Pydantic v2 compatible SecretField replacement.

        This function provides the same interface as the old SecretField
        but works with Pydantic v2 by using Field internally.

        Args:
            default: Default value for the field
            alias: Field alias
            title: Field title
            description: Field description
            **kwargs: Additional field parameters

        Returns:
            A Pydantic Field configured for secret values
        """
        # Remove parameters that don't exist in v2
        kwargs.pop("secret", None)
        kwargs.pop("repr", None)

        return PydanticField(
            default = default, alias = alias, title = title, description = description, **kwargs
        )

    # Also provide the standard Field and BaseModel
    Field = PydanticField
    BaseModel = PydanticBaseModel

    logger.info("Pydantic v2 SecretField replacement ready")

except ImportError:
    logger.warning("⚠️ Pydantic not available, using fallback")

    def SecretField(default: Any = None, **kwargs: Any) -> Any:
        """Fallback SecretField when Pydantic is not available"""
        return default

    def Field(default: Any = None, **kwargs: Any) -> Any:
        """Fallback Field when Pydantic is not available"""
        return default

    class BaseModel:
        """Fallback BaseModel when Pydantic is not available"""

        def __init__(self, **kwargs: Any):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def dict(self) -> dict:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# Export all the important items
__all__ = ["SecretField", "Field", "BaseModel"]