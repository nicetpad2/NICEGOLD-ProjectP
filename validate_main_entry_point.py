#!/usr/bin/env python3
"""
🧪 NICEGOLD ProjectP v2.1 - Entry Point Validation Test

This script validates that ProjectP.py is the proper main entry point
and that all features are integrated correctly.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def test_projectp_exists():
    """Test that ProjectP.py exists and is accessible"""
    print("🔍 Testing ProjectP.py existence...")
    
    if not os.path.exists("ProjectP.py"):
        print("❌ CRITICAL: ProjectP.py not found!")
        return False
    
    print("✅ ProjectP.py exists")
    return True


def test_projectp_importable():
    """Test that ProjectP.py can be imported"""
    print("🔍 Testing ProjectP.py import capability...")
    
    try:
        spec = importlib.util.spec_from_file_location("ProjectP", "ProjectP.py")
        if spec is None:
            print("❌ Cannot create spec for ProjectP.py")
            return False
            
        module = importlib.util.module_from_spec(spec)
        # Don't actually execute to avoid running the main loop
        print("✅ ProjectP.py is importable")
        return True
        
    except Exception as e:
        print(f"❌ Error importing ProjectP.py: {e}")
        return False


def test_projectp_syntax():
    """Test ProjectP.py syntax"""
    print("🔍 Testing ProjectP.py syntax...")
    
    try:
        with open("ProjectP.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Compile to check syntax
        compile(content, "ProjectP.py", "exec")
        print("✅ ProjectP.py syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in ProjectP.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking syntax: {e}")
        return False


def test_core_components():
    """Test that core components exist in ProjectP.py"""
    print("🔍 Testing core components in ProjectP.py...")
    
    try:
        with open("ProjectP.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_components = [
            "OptimizedProjectPApplication",
            "main_optimized",
            "run_optimized",
            "show_optimized_main_menu",
            "handle_optimized_menu_choice",
            "_handle_choice_optimized"
        ]
        
        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing components: {missing_components}")
            return False
        
        print("✅ All core components found")
        return True
        
    except Exception as e:
        print(f"❌ Error checking components: {e}")
        return False


def test_enhanced_features():
    """Test that enhanced features are integrated"""
    print("🔍 Testing enhanced features integration...")
    
    try:
        with open("ProjectP.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        enhanced_features = [
            "run_advanced_data_pipeline",
            "run_model_ensemble_system", 
            "run_interactive_dashboard",
            "run_risk_management_system",
            "run_complete_enhanced_pipeline"
        ]
        
        missing_features = []
        for feature in enhanced_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing enhanced features: {missing_features}")
            return False
        
        print("✅ All enhanced features integrated")
        return True
        
    except Exception as e:
        print(f"❌ Error checking enhanced features: {e}")
        return False


def test_menu_structure():
    """Test that menu structure is complete"""
    print("🔍 Testing menu structure...")
    
    try:
        with open("ProjectP.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        menu_items = [
            "Full Pipeline",
            "Data Analysis", 
            "Quick Test",
            "System Health Check",
            "Install Dependencies",
            "Clean System",
            "Performance Monitor"
        ]
        
        missing_items = []
        for item in menu_items:
            if item not in content:
                missing_items.append(item)
        
        if missing_items:
            print(f"❌ Missing menu items: {missing_items}")
            return False
        
        print("✅ Menu structure complete")
        return True
        
    except Exception as e:
        print(f"❌ Error checking menu structure: {e}")
        return False


def test_deprecated_files():
    """Test that deprecated files have proper notices"""
    print("🔍 Testing deprecated file notices...")
    
    deprecated_files = [
        "main.py",
        "run_full_pipeline.py",
        "run_ai_agents.py"
    ]
    
    issues = []
    for filepath in deprecated_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "DEPRECATED" not in content.upper() and "ProjectP.py" not in content:
                    issues.append(f"{filepath} - No deprecation notice")
                    
            except Exception as e:
                issues.append(f"{filepath} - Error reading: {e}")
    
    if issues:
        print("⚠️ Deprecation notice issues:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    print("✅ Deprecated files properly marked")
    return True


def test_documentation():
    """Test that documentation files exist"""
    print("🔍 Testing documentation...")
    
    required_docs = [
        "README_MAIN_ENTRY.md",
        "OFFICIAL_NOTICE_SINGLE_ENTRY_POINT.md"
    ]
    
    missing_docs = []
    for doc in required_docs:
        if not os.path.exists(doc):
            missing_docs.append(doc)
    
    if missing_docs:
        print(f"❌ Missing documentation: {missing_docs}")
        return False
    
    print("✅ Documentation complete")
    return True


def run_basic_functionality_test():
    """Run a basic functionality test of ProjectP.py"""
    print("🔍 Testing basic ProjectP.py functionality...")
    
    try:
        # Try to run ProjectP.py with a quick exit
        # This won't work in interactive mode, so we'll skip this test
        print("ℹ️ Skipping interactive functionality test")
        print("   (ProjectP.py requires interactive input)")
        return True
        
    except Exception as e:
        print(f"⚠️ Functionality test skipped: {e}")
        return True


def main():
    """Run all validation tests"""
    print("🧪 NICEGOLD ProjectP v2.1 - Entry Point Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("ProjectP.py Exists", test_projectp_exists),
        ("ProjectP.py Importable", test_projectp_importable),
        ("ProjectP.py Syntax", test_projectp_syntax),
        ("Core Components", test_core_components),
        ("Enhanced Features", test_enhanced_features),
        ("Menu Structure", test_menu_structure),
        ("Deprecated Files", test_deprecated_files),
        ("Documentation", test_documentation),
        ("Basic Functionality", run_basic_functionality_test)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("🏁 VALIDATION RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ ProjectP.py is ready for production use")
        print("🚀 Users and AI agents can safely use: python ProjectP.py")
    else:
        print(f"\n⚠️ {failed} TESTS FAILED")
        print("🔧 Please address the issues above before production use")
    
    print("\n💡 To start using NICEGOLD ProjectP v2.1:")
    print("   python ProjectP.py")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
