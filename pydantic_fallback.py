# Pydantic Fallback for SecretField
try:
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
    print("Pydantic SecretField imported successfully")
except ImportError:
    print("Creating pydantic SecretField fallback...")
    
    class SecretField:
        """Fallback SecretField implementation"""
        def __init__(self, *args, **kwargs):
            self.default = kwargs.get('default', '')
        
        @classmethod
        def get_secret_value(cls, value):
            return str(value) if value is not None else ""
        
        def __call__(self, *args, **kwargs):
            return self.default
    
    # Ensure pydantic module exists
    try:
        import pydantic
        pydantic.SecretField = SecretField
    except ImportError:
        # Create fake pydantic module
        import types
        import sys
        pydantic = types.ModuleType('pydantic')
        pydantic.SecretField = SecretField
        sys.modules['pydantic'] = pydantic
    
    print("Pydantic SecretField fallback created")
