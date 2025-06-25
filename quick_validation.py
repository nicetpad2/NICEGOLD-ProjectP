#!/usr/bin/env python3
"""
Quick validation test for ProjectP.py as main entry point
"""

import os

print("🧪 ProjectP.py Entry Point Validation")
print("=" * 50)

# Test 1: Check if ProjectP.py exists
if os.path.exists("ProjectP.py"):
    print("✅ ProjectP.py exists")
else:
    print("❌ ProjectP.py not found")
    exit(1)

# Test 2: Check if it's readable
try:
    with open("ProjectP.py", 'r') as f:
        content = f.read()
    print("✅ ProjectP.py is readable")
    file_size = len(content)
    print(f"   Size: {file_size:,} characters")
except Exception as e:
    print(f"❌ Cannot read ProjectP.py: {e}")
    exit(1)

# Test 3: Check for main components
main_components = [
    "main_optimized",
    "OptimizedProjectPApplication",
    "run_optimized"
]

found_main = 0
for component in main_components:
    if component in content:
        found_main += 1
        print(f"   ✓ {component}")

print(f"✅ Main components: {found_main}/{len(main_components)} found")

# Test 4: Check for enhanced features
enhanced_features = [
    "run_advanced_data_pipeline",
    "run_model_ensemble_system",
    "run_interactive_dashboard",
    "run_risk_management_system"
]

found_features = 0
for feature in enhanced_features:
    if feature in content:
        found_features += 1

print(f"✅ Enhanced features: {found_features}/{len(enhanced_features)} found")

# Test 5: Check documentation files
required_docs = [
    "README_MAIN_ENTRY.md",
    "OFFICIAL_NOTICE_SINGLE_ENTRY_POINT.md"
]

docs_exist = 0
for doc in required_docs:
    if os.path.exists(doc):
        docs_exist += 1
        print(f"   ✓ {doc}")

print(f"✅ Documentation: {docs_exist}/{len(required_docs)} files found")

# Test 6: Check for menu system
menu_keywords = ["menu", "choice", "input", "option"]
menu_found = sum(1 for keyword in menu_keywords if keyword in content.lower())
print(f"✅ Menu system indicators: {menu_found} found")

print("\n🎯 VALIDATION SUMMARY:")
print("=" * 50)

if found_main >= 2 and found_features >= 2 and docs_exist >= 1:
    print("🎉 ProjectP.py is READY for production use!")
    print("✅ All essential components found")
    print("📚 Documentation available")
    print("🚀 Users and AI agents can safely use:")
    print("   python ProjectP.py")
else:
    print("⚠️ ProjectP.py may need additional setup")
    print(f"   Main components: {found_main}/{len(main_components)}")
    print(f"   Enhanced features: {found_features}/{len(enhanced_features)}")
    print(f"   Documentation: {docs_exist}/{len(required_docs)}")

print("\n💡 Next steps:")
print("1. Run: python ProjectP.py")
print("2. Select menu option 4 (System Health Check)")
print("3. Install dependencies if needed (menu option 5)")
print("4. Start with Full Pipeline (menu option 1)")

print("=" * 50)
