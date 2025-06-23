#!/usr/bin/env python3
"""
Simple syntax test for ml.py import
ทดสอบ syntax error ใน ml.py
"""

print("🔧 Testing ml.py import...")

try:
    import src.features.ml
    print("✅ ml.py imported successfully!")
    
except SyntaxError as e:
    print(f"❌ SyntaxError in ml.py: {e}")
    print(f"   File: {e.filename}")
    print(f"   Line: {e.lineno}")
    print(f"   Text: {e.text}")
    
except Exception as e:
    print(f"❌ Other error: {e}")

print("🎉 Test completed!")
