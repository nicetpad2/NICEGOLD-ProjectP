"""
Test Pipeline Imports
====================
Simple test to verify the pipeline import fix works
"""

print("🧪 Testing pipeline import fix...")

# Initialize compatibility
import src.init_pipeline

print("\n📦 Testing SecretField import...")

try:
    from src.pydantic_secretfield import BaseModel, Field, SecretField

    print("✅ SUCCESS: SecretField imported successfully!")

    # Test SecretField
    field = SecretField(default="test_secret")
    print(f"✅ SUCCESS: SecretField created: {field}")

    # Test with a simple model
    class TestModel(BaseModel):
        secret_value: str = SecretField(default="secret")
        normal_value: str = Field(default="normal")

    model = TestModel()
    print(f"✅ SUCCESS: Model created with SecretField: {model.dict()}")

except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n📦 Testing alternative import...")

try:
    # Test using builtins
    SecretField = getattr(__builtins__, "SecretField", None)
    if SecretField:
        field = SecretField(default="test")
        print("✅ SUCCESS: Global SecretField works!")
    else:
        print("⚠️ WARNING: Global SecretField not available")

except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n🎉 Pipeline import test completed!")
