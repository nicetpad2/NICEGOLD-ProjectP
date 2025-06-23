#!/usr/bin/env python3
"""
ProjectP ML Protection Integration Demo
=====================================

This script demonstrates how to integrate the ML Protection System
with your ProjectP trading pipeline.
"""

import pandas as pd
import numpy as np
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def demonstrate_protection_integration():
    """Demonstrate the complete protection integration process"""
    
    print("🛡️ ProjectP ML Protection Integration Demo")
    print("=" * 60)
    
    # Step 1: Create realistic trading data
    print("\n📊 Step 1: Creating sample trading data...")
    
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Create realistic OHLCV data
    base_price = 100
    prices = [base_price]
    
    for i in range(1, n_samples):
        change = np.random.normal(0, 0.02)  # 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(1, new_price))  # Prevent negative prices
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 50000, n_samples),
    })
    
    # Add technical indicators
    data['sma_20'] = data['close'].rolling(20).mean()
    data['rsi'] = 50 + np.random.normal(0, 15, n_samples)  # Simplified RSI
    data['macd'] = np.random.normal(0, 0.1, n_samples)
    data['returns'] = data['close'].pct_change()
    
    # Create target (next-hour price direction)
    data['target'] = (data['returns'].shift(-1) > 0).astype(int)
    data = data.dropna()
    
    print(f"   ✅ Created {data.shape[0]} samples with {data.shape[1]} features")
    
    # Step 2: Basic Protection Analysis
    print("\n🔍 Step 2: Running basic protection analysis...")
    
    try:
        from ml_protection_system import MLProtectionSystem, ProtectionLevel
        
        # Initialize protection with enterprise level
        protection_system = MLProtectionSystem(ProtectionLevel.ENTERPRISE)
        
        # Run protection analysis
        result = protection_system.protect_dataset(
            data=data,
            target_col='target',
            timestamp_col='timestamp'
        )
        
        print(f"   ✅ Protection analysis complete!")
        print(f"      - Overall Clean: {result.is_clean}")
        print(f"      - Noise Score: {result.noise_score:.3f}")
        print(f"      - Leakage Score: {result.leakage_score:.3f}")
        print(f"      - Overfitting Score: {result.overfitting_score:.3f}")
        
        if result.issues_found:
            print(f"      - Issues Found: {len(result.issues_found)}")
            for issue in result.issues_found[:3]:  # Show first 3 issues
                print(f"        • {issue}")
        
        # Step 3: ProjectP Integration
        print("\n🔗 Step 3: Testing ProjectP integration...")
        
        try:
            from projectp_protection_integration import ProjectPProtectionIntegration
            
            # Initialize ProjectP integration
            integration = ProjectPProtectionIntegration(
                protection_level="enterprise",
                enable_tracking=True
            )
            
            # Test data pipeline protection
            protected_data = integration.protect_data_pipeline(
                data=data,
                target_col='target',
                timestamp_col='timestamp',
                stage="preprocessing"
            )
            
            print(f"   ✅ ProjectP integration working!")
            print(f"      - Original shape: {data.shape}")
            print(f"      - Protected shape: {protected_data.shape}")
            
            # Get protection summary
            summary = integration.get_protection_summary()
            if summary.get('total_protection_runs', 0) > 0:
                print(f"      - Protection runs: {summary['total_protection_runs']}")
                print(f"      - Latest status: {summary['latest_analysis']['is_clean']}")
            
        except ImportError as e:
            print(f"   ⚠️ ProjectP integration not available: {e}")
            
        # Step 4: Advanced Protection
        print("\n🛡️ Step 4: Testing advanced protection features...")
        
        try:
            from advanced_ml_protection_system import AdvancedMLProtectionSystem
            
            # Initialize advanced system
            advanced_system = AdvancedMLProtectionSystem()
            
            # Run comprehensive analysis
            report = advanced_system.analyze_data_comprehensive(
                data=data,
                target_column='target',
                feature_columns=[col for col in data.columns if col not in ['target', 'timestamp']]
            )
            
            print(f"   ✅ Advanced protection analysis complete!")
            print(f"      - Protection Score: {report.overall_protection_score:.3f}")
            print(f"      - Risk Level: {report.risk_level}")
            print(f"      - Data Quality Score: {report.data_quality_score:.3f}")
            
        except ImportError as e:
            print(f"   ⚠️ Advanced protection not available: {e}")
            
        # Step 5: Generate Report
        print("\n📄 Step 5: Generating protection report...")
        
        try:
            report_path = protection_system.generate_protection_report(result, "protection_demo_report.html")
            print(f"   ✅ Report generated: {report_path}")
        except Exception as e:
            print(f"   ⚠️ Report generation failed: {e}")
            
        return True
        
    except ImportError as e:
        print(f"   ❌ Core protection system not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error in protection analysis: {e}")
        return False

def show_next_steps():
    """Show next steps for integration"""
    
    print("\n🚀 Next Steps for ProjectP Integration:")
    print("=" * 60)
    
    print("\n1. 📋 CLI Usage:")
    print("   python advanced_ml_protection_cli.py analyze your_data.csv --target target")
    print("   python advanced_ml_protection_cli.py clean your_data.csv --output cleaned.csv")
    print("   python advanced_ml_protection_cli.py quick-check your_data.csv")
    
    print("\n2. 🔗 Direct Integration with ProjectP.py:")
    print("   # Add to your ProjectP.py file:")
    print("   from projectp_protection_integration import ProjectPProtectionIntegration")
    print("   protection = ProjectPProtectionIntegration()")
    print("   protected_data = protection.protect_data_pipeline(your_data)")
    
    print("\n3. 🛡️ Advanced Protection:")
    print("   # Use advanced features for enterprise-grade protection")
    print("   from advanced_ml_protection_system import AdvancedMLProtectionSystem")
    print("   system = AdvancedMLProtectionSystem()")
    print("   report = system.analyze_data_comprehensive(data, 'target')")
    
    print("\n4. 📊 Examples and Documentation:")
    print("   python ml_protection_examples.py              # Run examples")
    print("   python advanced_ml_protection_examples.py     # Advanced examples")
    print("   See: ADVANCED_ML_PROTECTION_COMPLETE_GUIDE.md # Complete guide")
    
    print("\n5. ⚙️ Configuration:")
    print("   Edit: advanced_ml_protection_config.yaml      # Main config")
    print("   Edit: ml_protection_config.yaml               # Basic config")

if __name__ == "__main__":
    success = demonstrate_protection_integration()
    
    if success:
        show_next_steps()
        print("\n✅ ML Protection System is ready for production use!")
    else:
        print("\n❌ Please check the error messages and ensure all files are present.")
        print("Required files:")
        print("  - ml_protection_system.py")
        print("  - projectp_protection_integration.py") 
        print("  - advanced_ml_protection_system.py")
        print("  - tracking.py")
