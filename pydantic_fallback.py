# Pydantic Fallback for SecretField
try:
    from pydantic import SecretField
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
