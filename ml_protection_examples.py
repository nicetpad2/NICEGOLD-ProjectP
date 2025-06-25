

# Demonstrates integration with ProjectP trading system
# Setup logging
# 🚀 Complete ML Protection System Example
from datetime import datetime, timedelta
    from ml_protection_system import MLProtectionSystem, ProtectionLevel
from pathlib import Path
    from projectp_protection_integration import ProjectPProtectionIntegration, apply_protection_to_existing_projectp
        from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from tracking import EnterpriseTracker
import logging
import numpy as np
import os
import pandas as pd
        import subprocess
import sys
        import traceback
import warnings
    import yaml
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    PROTECTION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Protection system not available: {e}")
    PROTECTION_AVAILABLE = False

def create_sample_trading_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create realistic sample trading data for demonstration"""

    # Generate timestamp sequence
    start_date = datetime.now() - timedelta(days = n_samples//24)
    timestamps = pd.date_range(start_date, periods = n_samples, freq = 'H')

    # Generate realistic trading features
    np.random.seed(42)

    # Price - related features
    price_base = 2000 + np.cumsum(np.random.randn(n_samples) * 0.5)

    data = {
        'timestamp': timestamps, 
        'open': price_base, 
        'high': price_base + np.abs(np.random.randn(n_samples) * 2), 
        'low': price_base - np.abs(np.random.randn(n_samples) * 2), 
        'close': price_base + np.random.randn(n_samples) * 1, 
        'volume': np.random.lognormal(10, 1, n_samples), 

        # Technical indicators
        'rsi': np.random.uniform(20, 80, n_samples), 
        'macd': np.random.randn(n_samples) * 5, 
        'bb_upper': price_base + np.random.uniform(10, 50, n_samples), 
        'bb_lower': price_base - np.random.uniform(10, 50, n_samples), 
        'atr': np.random.uniform(5, 25, n_samples), 

        # Market sentiment features
        'sentiment_score': np.random.uniform( - 1, 1, n_samples), 
        'news_impact': np.random.exponential(0.5, n_samples), 
        'volatility': np.random.uniform(0.1, 0.8, n_samples), 

        # Economic indicators
        'dollar_index': 100 + np.random.randn(n_samples) * 2, 
        'gold_etf_volume': np.random.lognormal(12, 0.5, n_samples), 
        'fed_rate_expectation': np.random.uniform(0, 5, n_samples), 
    }

    # Create target variable (1 for price increase, 0 for decrease)
    price_change = np.diff(price_base, prepend = price_base[0])
    data['target'] = (price_change > 0).astype(int)

    # Add some intentional issues for demonstration
    df = pd.DataFrame(data)

    # Add noise (outliers)
    noise_indices = np.random.choice(len(df), size = int(len(df) * 0.05), replace = False)
    df.loc[noise_indices, 'volume'] *= 100  # Extreme volume spikes

    # Add potential data leakage (future information)
    df['future_price_hint'] = df['close'].shift( - 1) + np.random.randn(n_samples) * 0.1

    # Add highly correlated feature (potential overfitting)
    df['target_noise'] = df['target'] + np.random.randn(n_samples) * 0.1

    # Add some missing values
    missing_indices = np.random.choice(len(df), size = int(len(df) * 0.02), replace = False)
    df.loc[missing_indices, 'sentiment_score'] = np.nan

    return df

def example_1_basic_protection():
    """Example 1: Basic protection analysis"""

    print("\n" + " = "*60)
    print("🛡️ Example 1: Basic Protection Analysis")
    print(" = "*60)

    if not PROTECTION_AVAILABLE:
        print("❌ Protection system not available")
        return

    # Create sample data
    data = create_sample_trading_data(500)
    print(f"📊 Created sample data: {data.shape}")

    # Initialize protection with basic level
    protection = ProjectPProtectionIntegration(protection_level = "basic")

    # Apply protection
    protected_data, summary = apply_protection_to_existing_projectp(
        data_df = data, 
        target_col = 'target', 
        timestamp_col = 'timestamp'
    )

    print(f"✅ Protection complete!")
    print(f"📈 Original shape: {data.shape}")
    print(f"📉 Protected shape: {protected_data.shape}")
    print(f"🎯 Summary: {summary['latest_analysis']}")

def example_2_enterprise_protection_with_model():
    """Example 2: Enterprise - level protection with model training"""

    print("\n" + " = "*60)
    print("🏢 Example 2: Enterprise Protection with Model Training")
    print(" = "*60)

    if not PROTECTION_AVAILABLE:
        print("❌ Protection system not available")
        return

    # Create larger dataset
    data = create_sample_trading_data(2000)
    print(f"📊 Created sample data: {data.shape}")

    # Initialize enterprise protection
    protection = ProjectPProtectionIntegration(protection_level = "enterprise")

    # Protect data pipeline
    protected_data = protection.protect_data_pipeline(
        data = data, 
        target_col = 'target', 
        timestamp_col = 'timestamp', 
        stage = "preprocessing"
    )

    # Prepare features and target
    feature_cols = [col for col in protected_data.columns
                   if col not in ['target', 'timestamp']]
    X = protected_data[feature_cols]
    y = protected_data['target']

    # Train a simple model for demonstration

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = RandomForestClassifier(n_estimators = 100, random_state = 42)

    # Apply model protection
    model_protection = protection.protect_model_training(
        X = X_train, y = y_train, model = model
    )

    print(f"🤖 Model protection result: {model_protection['should_train']}")

    if model_protection['should_train']:
        print("✅ Training approved - fitting model...")
        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print(f"📈 Train Score: {train_score:.4f}")
        print(f"📊 Test Score: {test_score:.4f}")

        # Generate comprehensive report
        report_path = protection.generate_protection_report()
        print(f"📄 Report generated: {report_path}")
    else:
        print("❌ Training blocked due to protection concerns!")
        print("Issues:", model_protection.get('issues_found', []))

def example_3_real_projectp_integration():
    """Example 3: Real integration with ProjectP pipeline"""

    print("\n" + " = "*60)
    print("🔗 Example 3: Real ProjectP Integration")
    print(" = "*60)

    if not PROTECTION_AVAILABLE:
        print("❌ Protection system not available")
        return

    # Simulate ProjectP workflow
    def simulate_projectp_pipeline():
        """Simulated ProjectP pipeline with protection"""

        # Step 1: Data Loading (with protection)
        print("📥 Step 1: Loading and protecting data...")
        data = create_sample_trading_data(1500)

        protection = ProjectPProtectionIntegration(protection_level = "enterprise")

        # Protect raw data
        protected_data = protection.protect_data_pipeline(
            data = data, 
            stage = "data_loading"
        )

        # Step 2: Feature Engineering (with protection)
        print("🔧 Step 2: Feature engineering with protection...")

        # Add some engineered features
        protected_data['price_momentum'] = protected_data['close'].pct_change(5)
        protected_data['volume_sma'] = protected_data['volume'].rolling(10).mean()
        protected_data['volatility_ratio'] = (
            protected_data['high'] - protected_data['low']
        ) / protected_data['close']

        # Protect engineered features
        final_data = protection.protect_data_pipeline(
            data = protected_data, 
            stage = "feature_engineering"
        )

        # Step 3: Model Training (with protection)
        print("🤖 Step 3: Model training with protection...")

        # Prepare final dataset
        feature_cols = [col for col in final_data.columns
                       if col not in ['target', 'timestamp']]
        X = final_data[feature_cols].fillna(0)  # Simple missing value handling
        y = final_data['target']

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Initialize model
        model = GradientBoostingClassifier(n_estimators = 100, random_state = 42)

        # Apply protection to training
        training_protection = protection.protect_model_training(
            X = X_train, y = y_train, model = model
        )

        if training_protection['should_train']:
            print("✅ Protection passed - training model...")
            model.fit(X_train, y_train)

            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            print(f"📊 Final Results:")
            print(f"  • Train Accuracy: {train_score:.4f}")
            print(f"  • Test Accuracy: {test_score:.4f}")
            print(f"  • Data Shape: {data.shape} → {final_data.shape}")

            # Generate comprehensive summary
            summary = protection.get_protection_summary()
            print(f"  • Protection Runs: {summary['total_protection_runs']}")
            print(f"  • Final Clean Status: {summary['latest_analysis']['is_clean']}")

            return model, final_data, summary
        else:
            print("❌ Training blocked - fix issues first!")
            return None, final_data, training_protection

    # Run the simulated pipeline
    result = simulate_projectp_pipeline()

    if result[0] is not None:
        print("🎉 ProjectP pipeline completed successfully with protection!")
    else:
        print("⚠️ Pipeline halted due to protection concerns.")

def example_4_cli_demonstration():
    """Example 4: Demonstrate CLI usage"""

    print("\n" + " = "*60)
    print("💻 Example 4: CLI Usage Demonstration")
    print(" = "*60)

    # Create sample data file
    data = create_sample_trading_data(800)
    sample_file = "sample_trading_data.csv"
    data.to_csv(sample_file, index = False)

    print(f"📁 Created sample file: {sample_file}")
    print("\n💡 CLI Commands you can try:")
    print(f"  1. Quick check:")
    print(f"     python ml_protection_cli.py quick - check {sample_file}")
    print(f"  2. Full analysis:")
    print(f"     python ml_protection_cli.py analyze -d {sample_file} -l enterprise - - report")
    print(f"  3. Data cleaning:")
    print(f"     python ml_protection_cli.py clean -d {sample_file} -o cleaned_data.csv")
    print(f"  4. Validation:")
    print(f"     python ml_protection_cli.py validate -d {sample_file}")
    print(f"  5. Initialize config:")
    print(f"     python ml_protection_cli.py init - config -l enterprise")

    # Try to run quick check if CLI is available
    try:
        result = subprocess.run([
            sys.executable, "ml_protection_cli.py", "quick - check", sample_file
        ], capture_output = True, text = True, timeout = 30)

        if result.returncode == 0:
            print(f"\n✅ CLI Quick Check Result:")
            print(result.stdout)
        else:
            print(f"\n⚠️ CLI Quick Check Failed:")
            print(result.stderr)
    except Exception as e:
        print(f"\n💡 Run CLI commands manually: {e}")

def example_5_advanced_configuration():
    """Example 5: Advanced configuration and customization"""

    print("\n" + " = "*60)
    print("⚙️ Example 5: Advanced Configuration")
    print(" = "*60)

    if not PROTECTION_AVAILABLE:
        print("❌ Protection system not available")
        return

    # Create custom configuration
    custom_config = {
        'protection': {
            'level': 'enterprise', 
            'enable_tracking': True, 
            'auto_fix_issues': True
        }, 
        'noise': {
            'contamination_rate': 0.05,  # Very strict
            'volatility_threshold': 1.0,  # Very sensitive
            'enable_adaptive_filtering': True
        }, 
        'leakage': {
            'temporal_gap_hours': 48,  # Large temporal gap
            'target_leakage_threshold': 0.3,  # Very strict
            'strict_time_validation': True
        }, 
        'overfitting': {
            'max_features_ratio': 0.1,  # Very few features
            'min_samples_per_feature': 50,  # Many samples per feature
            'cross_validation_folds': 10  # Thorough validation
        }
    }

    # Save custom config
    config_path = "custom_protection_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(custom_config, f, default_flow_style = False)

    print(f"💾 Created custom config: {config_path}")

    # Use custom configuration
    protection_system = MLProtectionSystem(
        protection_level = ProtectionLevel.ENTERPRISE, 
        config_path = config_path
    )

    # Test with sample data
    data = create_sample_trading_data(1000)
    result = protection_system.protect_dataset(
        data = data, 
        target_col = 'target', 
        timestamp_col = 'timestamp'
    )

    print(f"🎯 Custom Protection Results:")
    print(f"  • Noise Score: {result.noise_score:.4f}")
    print(f"  • Leakage Score: {result.leakage_score:.4f}")
    print(f"  • Overfitting Score: {result.overfitting_score:.4f}")
    print(f"  • Overall Clean: {result.is_clean}")

    if result.issues_found:
        print(f"⚠️ Issues with strict config:")
        for issue in result.issues_found[:3]:  # Show first 3
            print(f"  • {issue}")

def main():
    """Run all examples"""

    print("🛡️ ML Protection System - Complete Examples")
    print(" = "*60)
    print("This demonstrates advanced protection against:")
    print("  • 🔇 Noise and outliers")
    print("  • 💧 Data leakage")
    print("  • 📈 Overfitting")
    print(" = "*60)

    try:
        # Run all examples
        example_1_basic_protection()
        example_2_enterprise_protection_with_model()
        example_3_real_projectp_integration()
        example_4_cli_demonstration()
        example_5_advanced_configuration()

        print("\n" + " = "*60)
        print("🎉 All examples completed successfully!")
        print(" = "*60)

        print("\n💡 Next Steps:")
        print("1. Integrate with your existing ProjectP.py")
        print("2. Customize protection levels for your needs")
        print("3. Set up automated protection in your pipeline")
        print("4. Use CLI tools for batch processing")
        print("5. Monitor protection metrics in production")

        print("\n📚 Documentation:")
        print("• README.md - Complete system overview")
        print("• ml_protection_config.yaml - Configuration reference")
        print("• ml_protection_cli.py - Command - line interface")

    except Exception as e:
        logger.error(f"Error in examples: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()