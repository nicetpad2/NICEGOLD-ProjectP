#!/usr/bin/env python3
"""
Quick test of ML Protection System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_protection_system():
    """Test the ML Protection System"""
    print("🧪 Testing ML Protection System...")
    
    try:
        # Test core system
        from ml_protection_system import MLProtectionSystem, ProtectionLevel
        print("✅ Core ML Protection System imported successfully")
        
        # Test protection levels
        print("Available protection levels:")
        for level in ProtectionLevel:
            print(f"  - {level.value}")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        data = pd.DataFrame({
            'timestamp': dates,
            'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
            'volume': np.random.randint(1000, 10000, n_samples),
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
        })
        data['target'] = (data['close'].pct_change() > 0).astype(int)
        data = data.dropna()
        
        print(f"📊 Created test data: {data.shape}")
        
        # Initialize protection system
        protection_system = MLProtectionSystem(ProtectionLevel.STANDARD)
        print("✅ Protection system initialized")
        
        # Run protection analysis
        print("🔍 Running protection analysis...")
        result = protection_system.protect_dataset(
            data=data,
            target_col='target',
            timestamp_col='timestamp'
        )
        
        print(f"✅ Protection analysis complete!")
        print(f"   - Clean Status: {result.is_clean}")
        print(f"   - Noise Score: {result.noise_score:.3f}")
        print(f"   - Leakage Score: {result.leakage_score:.3f}")
        print(f"   - Overfitting Score: {result.overfitting_score:.3f}")
        print(f"   - Issues Found: {len(result.issues_found)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_projectp_integration():
    """Test ProjectP integration"""
    print("\n🔗 Testing ProjectP Integration...")
    
    try:
        from projectp_protection_integration import ProjectPProtectionIntegration
        print("✅ ProjectP Integration imported successfully")
        
        # Initialize integration
        integration = ProjectPProtectionIntegration(protection_level="standard")
        print("✅ ProjectP Integration initialized")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_advanced_protection():
    """Test Advanced Protection System"""
    print("\n🛡️ Testing Advanced Protection System...")
    
    try:
        from advanced_ml_protection_system import AdvancedMLProtectionSystem
        print("✅ Advanced Protection System imported successfully")
        
        # Initialize advanced system
        advanced_system = AdvancedMLProtectionSystem()
        print("✅ Advanced Protection System initialized")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ML PROTECTION SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Run tests
    test1 = test_protection_system()
    test2 = test_projectp_integration()
    test3 = test_advanced_protection()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Core Protection System: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"ProjectP Integration: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Advanced Protection: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 ALL TESTS PASSED! ML Protection System is ready!")
        print("\n📋 Next Steps:")
        print("1. Run protection examples: python ml_protection_examples.py")
        print("2. Use CLI: python advanced_ml_protection_cli.py --help")
        print("3. Integrate with ProjectP: See projectp_protection_integration.py")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
