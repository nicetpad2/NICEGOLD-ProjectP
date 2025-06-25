#!/usr/bin/env python3
"""
Test script for production feature engineering pipeline
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def create_sample_data():
    """Create sample financial data for testing"""
    # Generate 1000 data points spanning about 4 years (trading days)
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")

    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 100
    prices = []

    for i in range(1000):
        # Random walk with slight upward trend
        change = np.random.normal(
            0.001, 0.02
        )  # 0.1% average daily return, 2% volatility
        base_price *= 1 + change
        prices.append(base_price)

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i - 1] if i > 0 else price
        close_price = price
        volume = np.random.randint(100000, 1000000)

        data.append(
            {
                "timestamp": dates[i],
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close_price,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_production_features():
    """Test the ProductionFeatureEngineer"""
    print("Starting ProductionFeatureEngineer test...")

    try:
        # Import the feature engineer
        from src.production_features import ProductionFeatureEngineer

        print("✓ Successfully imported ProductionFeatureEngineer")

        # Create sample data
        print("Creating sample data...")
        df = create_sample_data()
        print(f"✓ Created sample data with {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Initialize feature engineer
        print("Initializing ProductionFeatureEngineer...")
        fe = ProductionFeatureEngineer()
        print("✓ Successfully initialized ProductionFeatureEngineer")

        # Test feature engineering
        print("Running feature engineering...")
        engineered_df = fe.engineer_features(df)
        print(f"✓ Feature engineering completed")
        print(f"  Input shape: {df.shape}")
        print(f"  Output shape: {engineered_df.shape}")
        print(f"  Added {engineered_df.shape[1] - df.shape[1]} new features")

        # Test feature summary
        print("Getting feature summary...")
        summary = fe.get_feature_summary()
        print(f"✓ Feature summary generated")
        print("Feature categories:")
        for category, count in summary.items():
            print(f"  {category}: {count} features")

        # Show sample of engineered features
        print("\nSample of engineered features:")
        feature_cols = [col for col in engineered_df.columns if col not in df.columns]
        print(f"New feature columns ({len(feature_cols)} total):")
        for i, col in enumerate(feature_cols[:10]):  # Show first 10
            print(f"  {i+1}. {col}")
        if len(feature_cols) > 10:
            print(f"  ... and {len(feature_cols) - 10} more")

        # Test data quality
        print("\nData quality check:")
        null_counts = engineered_df.isnull().sum().sum()
        inf_counts = (
            np.isinf(engineered_df.select_dtypes(include=[np.number])).sum().sum()
        )
        print(f"  Null values: {null_counts}")
        print(f"  Infinite values: {inf_counts}")

        if null_counts == 0 and inf_counts == 0:
            print("✓ Data quality check passed")
        else:
            print("⚠ Data quality issues detected")

        print("\n" + "=" * 60)
        print("PRODUCTION FEATURE ENGINEERING TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Error during feature engineering test:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Production Feature Engineering Test")
    print("=" * 60)

    success = test_production_features()

    if success:
        print(
            "\n🎉 All tests passed! The production feature engineering pipeline is working correctly."
        )
        sys.exit(0)
    else:
        print("\n💥 Tests failed. Please check the errors above.")
        sys.exit(1)
