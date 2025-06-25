#!/usr/bin/env python3
"""
Quick test for data validation with improved configuration
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.pipeline.data_validator import DataValidator

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get validation config
    validation_config = config.get('data_validation', {})
    print("🔧 Validation Configuration:")
    print(f"   • Outlier Rate Threshold: {validation_config.get('outlier_rate_threshold', 15)}%")
    print(f"   • Max Acceptable Gaps: {validation_config.get('max_acceptable_gaps', 1000)}")
    print(f"   • Max Gap Hours: {validation_config.get('max_gap_hours', 24)}h")
    
    # Initialize validator
    validator = DataValidator(validation_config)
    
    # Load sample data
    data_file = Path("datacsv/XAUUSD_M1.csv")
    if data_file.exists():
        print(f"\n📊 Testing with: {data_file.name}")
        df = pd.read_csv(data_file)
        print(f"   • Data shape: {df.shape}")
        
        # Run validation
        results = validator.validate_data(df)
        
        # Show results
        print(f"\n✅ Validation Status: {results['overall_status']}")
        
        # Check warnings
        warnings = results.get('warnings', [])
        if warnings:
            print(f"⚠️ Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"   • {warning}")
        else:
            print("✅ No warnings!")
            
        # Show outlier results
        outlier_results = results.get('outlier_analysis', {})
        if outlier_results:
            rate = outlier_results.get('outlier_rate', 0)
            count = outlier_results.get('outlier_count', 0)
            print(f"\n📈 Outlier Analysis:")
            print(f"   • Rate: {rate:.1f}%")
            print(f"   • Count: {count}")
        
        # Show gap results
        gap_results = results.get('gap_analysis', {})
        if gap_results:
            gap_count = gap_results.get('gap_count', 0)
            has_gaps = gap_results.get('has_gaps', False)
            print(f"\n⏰ Gap Analysis:")
            print(f"   • Has gaps: {has_gaps}")
            print(f"   • Gap count: {gap_count}")
            
    else:
        print("❌ No data file found for testing")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
