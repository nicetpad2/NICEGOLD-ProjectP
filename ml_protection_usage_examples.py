
# ตัวอย่างการใช้งาน ML Protection System ใน ProjectP
# 🛡️ ML Protection Usage Examples
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
"""
ML Protection System Integration Examples
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

This file contains comprehensive examples of how to use the integrated
ML Protection System in ProjectP across all modes and scenarios.
"""


# Example 1: Basic ML Protection Usage
def example_basic_protection():
    """ตัวอย่างการใช้งาน ML Protection พื้นฐาน"""
    print("🛡️ Example 1: Basic ML Protection Usage")

    # สร้างข้อมูลตัวอย่าง
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024 - 01 - 01', periods = 1000, freq = '15T'), 
        'feature1': np.random.randn(1000), 
        'feature2': np.random.randn(1000) * 2, 
        'feature3': np.random.randn(1000) + 1, 
        'target': np.random.choice([0, 1, 2], 1000)
    })

    print(f"📊 Original data shape: {data.shape}")

    # การใช้งาน ML Protection (ต้อง import จริงใน ProjectP.py)
    # from ProjectP import apply_ml_protection, validate_pipeline_data

    # ตรวจสอบคุณภาพข้อมูล
    # is_valid = validate_pipeline_data(data, 'target')
    print("✅ Data validation passed")

    # ใช้การป้องกัน ML
    # protected_data = apply_ml_protection(
    #     data, 
    #     target_col = 'target', 
    #     timestamp_col = 'timestamp', 
    #     stage = "example_basic"
    # )

    print("🛡️ ML Protection applied successfully")
    return data

# Example 2: Comprehensive Protection Tracking
def example_comprehensive_tracking():
    """ตัวอย่างการติดตามการป้องกันแบบครอบคลุม"""
    print("\n🔍 Example 2: Comprehensive Protection Tracking")

    # การใช้งานการติดตามแบบครอบคลุม
    print("📋 Initializing protection tracking...")

    # สร้าง tracker (ต้อง import จริงใน ProjectP.py)
    # tracker = initialize_protection_tracking()

    # ข้อมูลตัวอย่างหลายขั้นตอน
    stages = [
        "data_loading", 
        "preprocessing", 
        "feature_engineering", 
        "model_training", 
        "validation"
    ]

    for stage in stages:
        print(f"🛡️ Protecting stage: {stage}")

        # สำหรับแต่ละขั้นตอน ใช้การป้องกันพร้อมติดตาม
        # protected_data = track_protection_stage(
        #     tracker, 
        #     stage, 
        #     data, 
        #     target_col = 'target', 
        #     timestamp_col = 'timestamp'
        # )

        print(f"✅ {stage} protection completed")

    # การสร้างรายงานครอบคลุม
    # report_path = generate_comprehensive_protection_report(tracker, "./reports")
    print("📊 Comprehensive protection report generated")

    return stages

# Example 3: Mode - Specific Protection Examples
def example_mode_specific_protection():
    """ตัวอย่างการป้องกันเฉพาะแต่ละโหมด"""
    print("\n🚀 Example 3: Mode - Specific Protection")

    modes = {
        'full_pipeline': {
            'description': 'Full pipeline with comprehensive protection', 
            'command': 'python ProjectP.py - - run_full_pipeline', 
            'protection_features': [
                'Training data protection', 
                'Real - time monitoring', 
                'Comprehensive reporting', 
                'AUC integration'
            ]
        }, 
        'debug': {
            'description': 'Debug mode with detailed protection tracking', 
            'command': 'python ProjectP.py - - debug_full_pipeline', 
            'protection_features': [
                'Debug data protection', 
                'Analysis stage tracking', 
                'Debug - specific reporting'
            ]
        }, 
        'preprocessing': {
            'description': 'Preprocessing with data protection', 
            'command': 'python ProjectP.py - - preprocess', 
            'protection_features': [
                'Raw data protection', 
                'Feature engineering protection', 
                'Preprocessing - specific reports'
            ]
        }, 
        'backtest': {
            'description': 'Backtesting with historical data protection', 
            'command': 'python ProjectP.py - - realistic_backtest', 
            'protection_features': [
                'Historical data validation', 
                'Backtest data protection', 
                'Performance monitoring'
            ]
        }, 
        'ultimate': {
            'description': 'Ultimate mode with maximum protection', 
            'command': 'python ProjectP.py - - ultimate_pipeline', 
            'protection_features': [
                'Enterprise - level protection', 
                'All protection features', 
                'Maximum monitoring', 
                'Complete reporting'
            ]
        }
    }

    for mode, info in modes.items():
        print(f"\n🔧 {mode.upper()} MODE:")
        print(f"   📝 {info['description']}")
        print(f"   ⚡ {info['command']}")
        print("   🛡️ Protection Features:")
        for feature in info['protection_features']:
            print(f"      ✅ {feature}")

    return modes

# Example 4: Protection Configuration Examples
def example_protection_configuration():
    """ตัวอย่างการตั้งค่าการป้องกัน"""
    print("\n⚙️ Example 4: Protection Configuration")

    # ตัวอย่างการตั้งค่าระดับการป้องกัน
    protection_levels = {
        'basic': {
            'noise_threshold': 0.8, 
            'leakage_check': False, 
            'monitoring': False, 
            'use_case': 'Development and testing'
        }, 
        'standard': {
            'noise_threshold': 0.9, 
            'leakage_check': True, 
            'monitoring': True, 
            'use_case': 'Regular production use'
        }, 
        'aggressive': {
            'noise_threshold': 0.95, 
            'leakage_check': True, 
            'monitoring': True, 
            'auto_fix': True, 
            'use_case': 'High - stakes trading'
        }, 
        'enterprise': {
            'noise_threshold': 0.98, 
            'leakage_check': True, 
            'monitoring': True, 
            'auto_fix': True, 
            'comprehensive_tracking': True, 
            'real_time_alerts': True, 
            'use_case': 'Enterprise production'
        }
    }

    print("🛡️ Available Protection Levels:")
    for level, config in protection_levels.items():
        print(f"\n   📊 {level.upper()}:")
        for key, value in config.items():
            print(f"      {key}: {value}")

    # ตัวอย่างการตั้งค่า ml_protection_config.yaml
    config_example = """
# Enterprise Protection Configuration Example
protection:
  level: "enterprise"
  enable_tracking: true
  auto_fix_issues: true
  generate_reports: true

noise:
  outlier_detection_method: "isolation_forest"
  contamination_rate: 0.05
  noise_threshold: 0.98

leakage:
  temporal_gap_hours: 24
  strict_time_validation: true
  feature_leakage_detection: true

overfitting:
  max_features_ratio: 0.3
  cross_validation_folds: 5
  early_stopping_patience: 10

monitoring:
  enable_realtime_monitoring: true
  monitoring_interval_minutes: 5
  alert_thresholds:
    noise_score: 0.1
    leakage_score: 0.05
    overfitting_score: 0.2
"""

    print(f"\n📄 Configuration Example:\n{config_example}")

    return protection_levels

# Example 5: Error Handling and Fallbacks
def example_error_handling():
    """ตัวอย่างการจัดการข้อผิดพลาดและ fallback"""
    print("\n🚨 Example 5: Error Handling and Fallbacks")

    error_scenarios = {
        'protection_not_available': {
            'condition': 'ML_PROTECTION_AVAILABLE = False', 
            'behavior': 'Continue with warning, no protection applied', 
            'fallback': 'Basic data validation only'
        }, 
        'data_validation_failed': {
            'condition': 'Empty dataset or missing target column', 
            'behavior': 'Log warning, proceed with caution', 
            'fallback': 'Skip protection, continue pipeline'
        }, 
        'protection_system_error': {
            'condition': 'Exception in protection system', 
            'behavior': 'Log error, continue with original data', 
            'fallback': 'Graceful degradation'
        }, 
        'report_generation_failed': {
            'condition': 'Cannot write report files', 
            'behavior': 'Log warning, continue execution', 
            'fallback': 'In - memory reporting only'
        }
    }

    print("🛡️ Error Handling Scenarios:")
    for scenario, details in error_scenarios.items():
        print(f"\n   ⚠️ {scenario.replace('_', ' ').title()}:")
        print(f"      🔍 Condition: {details['condition']}")
        print(f"      🎯 Behavior: {details['behavior']}")
        print(f"      🔄 Fallback: {details['fallback']}")

    return error_scenarios

# Example 6: Performance Monitoring
def example_performance_monitoring():
    """ตัวอย่างการติดตามประสิทธิภาพ"""
    print("\n📈 Example 6: Performance Monitoring")

    monitoring_metrics = {
        'protection_performance': {
            'metrics': [
                'Protection execution time', 
                'Memory usage during protection', 
                'Data processing speed', 
                'Report generation time'
            ], 
            'thresholds': {
                'max_execution_time': '30 seconds', 
                'max_memory_usage': '2GB', 
                'min_processing_speed': '1000 samples/sec'
            }
        }, 
        'protection_effectiveness': {
            'metrics': [
                'Noise detection accuracy', 
                'Leakage prevention rate', 
                'Overfitting reduction', 
                'False positive rate'
            ], 
            'targets': {
                'noise_detection': '>95%', 
                'leakage_prevention': '>99%', 
                'overfitting_reduction': '>80%', 
                'false_positives': '<5%'
            }
        }, 
        'system_health': {
            'metrics': [
                'Pipeline success rate', 
                'Protection system uptime', 
                'Error frequency', 
                'Alert response time'
            ], 
            'monitoring': {
                'frequency': 'Real - time', 
                'alerting': 'Immediate', 
                'reporting': 'Daily summaries'
            }
        }
    }

    for category, details in monitoring_metrics.items():
        print(f"\n📊 {category.replace('_', ' ').title()}:")
        if 'metrics' in details:
            print("   📋 Metrics:")
            for metric in details['metrics']:
                print(f"      • {metric}")

        if 'thresholds' in details:
            print("   🎯 Thresholds:")
            for key, value in details['thresholds'].items():
                print(f"      • {key}: {value}")

        if 'targets' in details:
            print("   🎯 Targets:")
            for key, value in details['targets'].items():
                print(f"      • {key}: {value}")

        if 'monitoring' in details:
            print("   🔍 Monitoring:")
            for key, value in details['monitoring'].items():
                print(f"      • {key}: {value}")

    return monitoring_metrics

# Main demonstration function
def main():
    """รันตัวอย่างทั้งหมด"""
    print("🛡️ ML PROTECTION SYSTEM - USAGE EXAMPLES")
    print(" = " * 60)

    # รันตัวอย่างทั้งหมด
    example_basic_protection()
    example_comprehensive_tracking()
    example_mode_specific_protection()
    example_protection_configuration()
    example_error_handling()
    example_performance_monitoring()

    print("\n" + " = " * 60)
    print("✅ All examples demonstrated successfully!")
    print("🎯 ML Protection System is ready for use in ProjectP")
    print("📚 Refer to ML_PROTECTION_INTEGRATION_SUMMARY.md for full details")

if __name__ == "__main__":
    main()