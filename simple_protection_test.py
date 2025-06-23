"""
Simple ML Protection Test
"""

def test_imports():
    """Test all protection system imports"""
    
    print("Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ pandas, numpy OK")
    except ImportError as e:
        print(f"❌ Basic imports failed: {e}")
        return False
    
    try:
        from ml_protection_system import MLProtectionSystem, ProtectionLevel
        print("✅ ml_protection_system OK")
    except ImportError as e:
        print(f"❌ ml_protection_system import failed: {e}")
        print("Make sure ml_protection_system.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ ml_protection_system error: {e}")
        return False
    
    try:
        from projectp_protection_integration import ProjectPProtectionIntegration
        print("✅ projectp_protection_integration OK")
    except ImportError as e:
        print(f"❌ projectp_protection_integration import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ projectp_protection_integration error: {e}")
        return False
    
    try:
        from advanced_ml_protection_system import AdvancedMLProtectionSystem
        print("✅ advanced_ml_protection_system OK")
    except ImportError as e:
        print(f"❌ advanced_ml_protection_system import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ advanced_ml_protection_system error: {e}")
        return False
    
    print("✅ All imports successful!")
    return True

def test_basic_functionality():
    """Test basic functionality"""
    
    print("\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from ml_protection_system import MLProtectionSystem, ProtectionLevel
        
        # Create simple test data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Initialize system
        system = MLProtectionSystem(ProtectionLevel.BASIC)
        print("✅ System initialized")
        
        # Test analysis
        result = system.protect_dataset(data, 'target', 'timestamp')
        print(f"✅ Analysis complete - Clean: {result.is_clean}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 SIMPLE ML PROTECTION TEST")
    print("=" * 50)
    
    test1 = test_imports()
    if test1:
        test2 = test_basic_functionality()
        
        if test2:
            print("\n🎉 SUCCESS! ML Protection System is working!")
            print("\nYou can now:")
            print("1. Run: python ml_protection_examples.py")
            print("2. Run: python advanced_ml_protection_examples.py") 
            print("3. Use CLI: python advanced_ml_protection_cli.py --help")
            print("4. Integrate with your ProjectP pipeline")
        else:
            print("\n⚠️ Basic functionality test failed")
    else:
        print("\n❌ Import tests failed - check file locations")
