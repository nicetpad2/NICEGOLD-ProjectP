#!/usr/bin/env python3
from ml_protection_system import MLProtectionSystem, ProtectionLevel
from projectp_protection_integration import ProjectPProtectionIntegration
import pandas as pd
"""
🛡️ ML Protection System - Quick Start Guide
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Your ML Protection System is comprehensive and production - ready!
This guide shows you how to use it effectively.
"""

def show_system_overview():
    """Show overview of the ML Protection System"""
    print("🛡️ ML PROTECTION SYSTEM OVERVIEW")
    print(" = " * 60)

    print("\n📦 Core Components Available:")
    print("✅ ml_protection_system.py          - Core protection engine")
    print("✅ projectp_protection_integration.py - ProjectP integration")
    print("✅ advanced_ml_protection_system.py  - Advanced features")
    print("✅ advanced_ml_protection_cli.py     - Command - line interface")
    print("✅ tracking.py                       - Experiment tracking")
    print("✅ Configuration files               - YAML configs")
    print("✅ Examples and documentation        - Complete guides")

def show_quick_usage_examples():
    """Show quick usage examples"""
    print("\n🚀 QUICK USAGE EXAMPLES")
    print(" = " * 60)

    print("\n1️⃣ Basic Python API Usage:")
    print("""

# Load your trading data
data = pd.read_csv('your_trading_data.csv')

# Initialize protection system
protection = MLProtectionSystem(ProtectionLevel.ENTERPRISE)

# Analyze and protect your data
result = protection.protect_dataset(
    data = data, 
    target_col = 'target', 
    timestamp_col = 'timestamp'
)

# Check results
print(f"Data is clean: {result.is_clean}")
print(f"Noise score: {result.noise_score:.3f}")
print(f"Issues found: {len(result.issues_found)}")
""")

    print("\n2️⃣ ProjectP Integration:")
    print("""

# Initialize ProjectP integration
integration = ProjectPProtectionIntegration(
    protection_level = "enterprise", 
    enable_tracking = True
)

# Protect your ProjectP data pipeline
protected_data = integration.protect_data_pipeline(
    data = your_trading_data, 
    target_col = 'target', 
    stage = "preprocessing"
)

# Protect model training
training_result = integration.protect_model_training(
    X = features, y = target, model = your_model
)

if training_result['should_train']:
    # Safe to train the model
    your_model.fit(features, target)
""")

    print("\n3️⃣ CLI Usage:")
    print("""
# Analyze data quality
python advanced_ml_protection_cli.py analyze data.csv - - target target

# Clean data automatically
python advanced_ml_protection_cli.py clean data.csv - - output cleaned.csv

# Quick health check
python advanced_ml_protection_cli.py quick - check data.csv

# Generate configuration
python advanced_ml_protection_cli.py config - - template trading

# ProjectP integration setup
python advanced_ml_protection_cli.py projectp - integrate data.csv
""")

def show_advanced_features():
    """Show advanced features"""
    print("\n🔬 ADVANCED FEATURES")
    print(" = " * 60)

    print("\n🛡️ Protection Capabilities:")
    print("• Noise detection and cleaning")
    print("• Data leakage prevention")
    print("• Overfitting protection")
    print("• Feature selection validation")
    print("• Time series cross - validation")
    print("• Market regime awareness")
    print("• Real - time monitoring")
    print("• Automated remediation")

    print("\n📊 Protection Levels:")
    print("• BASIC     - Essential protection")
    print("• STANDARD  - Balanced protection")
    print("• AGGRESSIVE- Strict protection")
    print("• ENTERPRISE- Maximum protection")

    print("\n🔧 Integration Options:")
    print("• Direct Python API")
    print("• Command - line interface")
    print("• ProjectP pipeline integration")
    print("• Jupyter notebook support")
    print("• Automated CI/CD integration")

def show_configuration_options():
    """Show configuration options"""
    print("\n⚙️ CONFIGURATION")
    print(" = " * 60)

    print("\n📁 Configuration Files:")
    print("• advanced_ml_protection_config.yaml - Main config")
    print("• ml_protection_config.yaml         - Basic config")
    print("• config.yaml                       - ProjectP config")

    print("\n🎛️ Key Configuration Options:")
    print("• Data quality thresholds")
    print("• Protection sensitivity levels")
    print("• Feature selection parameters")
    print("• Cross - validation settings")
    print("• Monitoring and alerting")
    print("• Trading - specific settings")

def show_next_steps():
    """Show recommended next steps"""
    print("\n📋 RECOMMENDED NEXT STEPS")
    print(" = " * 60)

    print("\n1️⃣ Test the System:")
    print("   python ml_protection_examples.py")
    print("   python advanced_ml_protection_examples.py")

    print("\n2️⃣ CLI Exploration:")
    print("   python advanced_ml_protection_cli.py - - help")
    print("   python advanced_ml_protection_cli.py status")

    print("\n3️⃣ Integrate with ProjectP:")
    print("   • Edit your main ProjectP.py file")
    print("   • Add protection calls to your pipeline")
    print("   • Configure protection levels")
    print("   • Set up monitoring")

    print("\n4️⃣ Customize Configuration:")
    print("   • Edit protection config files")
    print("   • Set trading - specific parameters")
    print("   • Configure alerts and thresholds")

    print("\n5️⃣ Production Deployment:")
    print("   • Set up automated protection")
    print("   • Configure monitoring dashboards")
    print("   • Implement alert systems")
    print("   • Document your protection strategy")

def show_integration_code():
    """Show integration code for ProjectP"""
    print("\n🔗 PROJECTP INTEGRATION CODE")
    print(" = " * 60)

    print("\nAdd this to your ProjectP.py file:")
    print("""
# At the top of ProjectP.py

class ProjectPWithProtection:
    def __init__(self):
        # Initialize protection
        self.protection = ProjectPProtectionIntegration(
            protection_level = "enterprise", 
            enable_tracking = True
        )

    def load_and_protect_data(self, data_path):
        \"\"\"Load and protect trading data\"\"\"
        # Load data
        data = pd.read_csv(data_path)

        # Apply protection
        protected_data = self.protection.protect_data_pipeline(
            data = data, 
            target_col = 'target', 
            timestamp_col = 'timestamp', 
            stage = "data_loading"
        )

        return protected_data

    def train_protected_model(self, X, y, model):
        \"\"\"Train model with protection\"\"\"
        # Check if training should proceed
        training_result = self.protection.protect_model_training(
            X = X, y = y, model = model
        )

        if training_result['should_train']:
            model.fit(X, y)
            print("✅ Model trained successfully")
        else:
            print("❌ Training blocked due to protection concerns")
            print("Issues:", training_result['issues_found'])

        return training_result
""")

def main():
    """Main function to show the complete guide"""
    print("🛡️ ML PROTECTION SYSTEM - COMPLETE GUIDE")
    print(" = " * 80)

    show_system_overview()
    show_quick_usage_examples()
    show_advanced_features()
    show_configuration_options()
    show_integration_code()
    show_next_steps()

    print("\n✅ YOUR ML PROTECTION SYSTEM IS READY!")
    print("🎯 Focus on integrating with your ProjectP pipeline")
    print("📚 Refer to the complete documentation for detailed usage")
    print("🔧 Customize configurations for your specific needs")

if __name__ == "__main__":
    main()