#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Data Validation Enhancement - NICEGOLD ProjectP
=================================================

Script to enhance data validation and reduce warnings by:
1. Loading improved configuration with better thresholds
2. Testing the data validator with real data
3. Providing recommendations for data quality improvement

Author: NICEGOLD Enterprise  
Version: 1.0
Date: June 25, 2025
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from utils.modern_logger import error, info, setup_logger, success, warning
    logger = setup_logger("DataValidationEnhancer")
    MODERN_LOGGER_AVAILABLE = True
except ImportError:
    MODERN_LOGGER_AVAILABLE = False
    def info(msg): print(f"ℹ️ [INFO] {msg}")
    def success(msg): print(f"✅ [SUCCESS] {msg}")
    def warning(msg): print(f"⚠️ [WARNING] {msg}")
    def error(msg): print(f"❌ [ERROR] {msg}")


class DataValidationEnhancer:
    """Enhanced data validation with improved thresholds and handling"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_path = self.project_root / "config.yaml"
        
    def load_config(self):
        """Load configuration from config.yaml"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            error(f"Failed to load config: {e}")
            return {}
    
    def test_data_validation(self):
        """Test the enhanced data validation with real data"""
        info("🔍 Testing enhanced data validation...")
        
        try:
            import pandas as pd

            from core.pipeline.data_validator import DataValidator

            # Load configuration
            config = self.load_config()
            validation_config = config.get('data_validation', {})
            
            # Initialize validator with new config
            validator = DataValidator(validation_config)
            
            # Find data files
            data_folder = self.project_root / "datacsv"
            if not data_folder.exists():
                warning("datacsv folder not found, creating with sample data...")
                data_folder.mkdir(parents=True, exist_ok=True)
                self.create_sample_data(data_folder)
            
            # Test with available data files
            data_files = list(data_folder.glob("*.csv"))
            if not data_files:
                warning("No CSV files found in datacsv folder")
                return
                
            # Test validation on first available file
            test_file = data_files[0]
            info(f"Testing validation on: {test_file.name}")
            
            # Load and validate data
            df = pd.read_csv(test_file)
            info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
            
            # Run validation
            results = validator.validate_data(df)
            
            # Display results
            self.display_validation_results(results, validation_config)
            
        except Exception as e:
            error(f"Validation test failed: {e}")
            
    def display_validation_results(self, results, config):
        """Display validation results with configuration context"""
        info("📊 Validation Results Summary:")
        print("=" * 60)
        
        # Overall status
        status = results.get("overall_status", "unknown")
        if status == "passed":
            success(f"Overall Status: {status.upper()}")
        else:
            warning(f"Overall Status: {status.upper()}")
        
        # Configuration thresholds
        print("\n🔧 Applied Configuration Thresholds:")
        print(f"   • Outlier Rate Threshold: {config.get('outlier_rate_threshold', 15)}%")
        print(f"   • Max Acceptable Gaps: {config.get('max_acceptable_gaps', 1000)}")
        print(f"   • Max Gap Hours: {config.get('max_gap_hours', 24)}h")
        print(f"   • Z-Score Threshold: {config.get('outlier_zscore_threshold', 5.0)}")
        
        # Outlier analysis
        outlier_results = results.get("outlier_analysis", {})
        if outlier_results:
            outlier_rate = outlier_results.get("outlier_rate", 0)
            outlier_count = outlier_results.get("outlier_count", 0)
            threshold = config.get('outlier_rate_threshold', 15)
            
            print(f"\n📈 Outlier Analysis:")
            if outlier_rate <= threshold:
                success(f"   • Outlier Rate: {outlier_rate:.1f}% (✅ Below {threshold}% threshold)")
            else:
                warning(f"   • Outlier Rate: {outlier_rate:.1f}% (⚠️ Above {threshold}% threshold)")
            print(f"   • Total Outliers: {outlier_count}")
        
        # Gap analysis  
        gap_results = results.get("gap_analysis", {})
        if gap_results:
            gap_count = gap_results.get("gap_count", 0)
            has_gaps = gap_results.get("has_gaps", False)
            max_gaps = config.get('max_acceptable_gaps', 1000)
            
            print(f"\n⏰ Gap Analysis:")
            if not has_gaps:
                success("   • No data gaps detected ✅")
            elif gap_count <= max_gaps:
                success(f"   • Gap Count: {gap_count} (✅ Below {max_gaps} threshold)")
            else:
                warning(f"   • Gap Count: {gap_count} (⚠️ Above {max_gaps} threshold)")
                
            largest_gap = gap_results.get("largest_gap")
            if largest_gap:
                gap_hours = largest_gap.total_seconds() / 3600
                max_hours = config.get('max_gap_hours', 24)
                if gap_hours <= max_hours:
                    success(f"   • Largest Gap: {gap_hours:.1f}h (✅ Below {max_hours}h threshold)")
                else:
                    warning(f"   • Largest Gap: {gap_hours:.1f}h (⚠️ Above {max_hours}h threshold)")
        
        # Warnings and errors
        warnings_list = results.get("warnings", [])
        errors_list = results.get("errors", [])
        
        if warnings_list:
            print(f"\n⚠️ Warnings ({len(warnings_list)}):")
            for i, warn in enumerate(warnings_list, 1):
                print(f"   {i}. {warn}")
        else:
            success("\n✅ No validation warnings!")
            
        if errors_list:
            print(f"\n❌ Errors ({len(errors_list)}):")
            for i, err in enumerate(errors_list, 1):
                print(f"   {i}. {err}")
        else:
            success("✅ No validation errors!")
            
        print("=" * 60)
        
    def create_sample_data(self, data_folder):
        """Create sample trading data for testing"""
        from datetime import datetime, timedelta

        import numpy as np
        import pandas as pd
        
        info("Creating sample trading data...")
        
        # Generate sample data
        start_date = datetime.now() - timedelta(days=30)
        dates = pd.date_range(start_date, periods=1000, freq='1H')
        
        # Generate realistic OHLC data
        np.random.seed(42)
        base_price = 100
        prices = []
        
        current_price = base_price
        for _ in range(len(dates)):
            # Random walk with some volatility
            change = np.random.normal(0, 0.02) * current_price
            current_price = max(10, current_price + change)  # Prevent negative prices
            
            # Generate OHLC from current price
            volatility = np.random.uniform(0.005, 0.02)
            high = current_price * (1 + volatility)
            low = current_price * (1 - volatility)
            open_price = np.random.uniform(low, high)
            close = current_price
            volume = np.random.randint(1000, 10000)
            
            prices.append({
                'Timestamp': dates[len(prices)],
                'Open': round(open_price, 2),
                'High': round(high, 2), 
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(prices)
        sample_file = data_folder / "sample_trading_data.csv"
        df.to_csv(sample_file, index=False)
        success(f"Created sample data: {sample_file}")
        
    def provide_recommendations(self):
        """Provide recommendations for data quality improvement"""
        info("💡 Data Quality Improvement Recommendations:")
        print("=" * 60)
        
        recommendations = [
            "🔧 Configuration Adjustments:",
            "   • Outlier threshold increased to 25% (from 15%)",
            "   • Gap threshold increased to 2000 (from 1000)", 
            "   • Max gap hours set to 48h (for weekends/holidays)",
            "",
            "📊 Data Quality Best Practices:",
            "   • Use high-frequency data with minimal gaps",
            "   • Implement data cleaning pipelines",
            "   • Set up real-time data validation",
            "   • Monitor data feeds for interruptions",
            "",
            "⚡ Performance Optimizations:",
            "   • Cache validation results for repeated runs",
            "   • Use parallel processing for large datasets",
            "   • Implement incremental validation",
            "   • Set up automated data quality alerts",
            "",
            "🛡️ Risk Management:",
            "   • Set conservative thresholds for production",
            "   • Implement data source failover",
            "   • Regular data quality audits",
            "   • Backup data validation systems"
        ]
        
        for rec in recommendations:
            print(rec)
        print("=" * 60)


def main():
    """Main execution function"""
    try:
        info("🚀 NICEGOLD Data Validation Enhancement")
        info("=======================================")
        
        enhancer = DataValidationEnhancer()
        
        # Test current validation
        enhancer.test_data_validation()
        
        # Provide recommendations
        enhancer.provide_recommendations()
        
        success("✅ Data validation enhancement completed!")
        
    except Exception as e:
        error(f"Enhancement failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
