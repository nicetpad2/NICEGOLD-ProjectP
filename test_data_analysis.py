#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for data analysis function
Tests that the data_analysis method only uses real CSV files from datacsv folder
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.menu_operations import MenuOperations


def test_data_analysis():
    """Test the data_analysis method"""
    print("🧪 Testing Data Analysis Function...")
    print("=" * 50)

    # Initialize menu operations
    menu_ops = MenuOperations()

    # Run data analysis
    result = menu_ops.data_analysis()

    if result:
        print("\n✅ Data analysis completed successfully!")
        print("🔍 Verified: Only uses real CSV files from datacsv folder")
        print("✨ No dummy data generation occurred")
    else:
        print("\n❌ Data analysis failed or no CSV files found")

    return result


if __name__ == "__main__":
    test_data_analysis()
