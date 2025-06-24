"""
Real Data Configuration for ProjectP
Ensures all pipeline modes use only real data from datacsv folder
"""

import os
from pathlib import Path

def get_real_data_config():
    """
    Get configuration that enforces real data usage from datacsv folder
    """
    # Get the datacsv folder path
    base_dir = Path(__file__).parent
    datacsv_path = base_dir / "projectp" / "datacsv"
    
    # Ensure datacsv exists
    if not datacsv_path.exists():
        raise FileNotFoundError(f"❌ CRITICAL: datacsv folder not found at {datacsv_path}")
    
    # Get available data files
    csv_files = list(datacsv_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"❌ CRITICAL: No CSV files found in datacsv folder")
    
    # Default to first available file
    default_file = csv_files[0].name
    
    return {
        "data": {
            "enforce_real_data": True,
            "datacsv_path": str(datacsv_path),
            "file": default_file,  # Default file from datacsv
            "available_files": [f.name for f in csv_files],
            "multi": False,  # Set to True to use all files
            "validation": {
                "min_rows": 100,
                "required_columns": ["Open", "High", "Low", "Close"],
                "no_dummy_data": True,
                "no_synthetic_data": True
            }
        },
        "pipeline": {
            "halt_on_missing_data": True,
            "halt_on_dummy_data": True,
            "require_datacsv": True
        }
    }

def validate_real_data_config(config):
    """
    Validate that configuration enforces real data usage
    """
    if not config.get("data", {}).get("enforce_real_data"):
        raise ValueError("❌ CRITICAL: Configuration must enforce real data usage")
    
    datacsv_path = config.get("data", {}).get("datacsv_path")
    if not datacsv_path or not os.path.exists(datacsv_path):
        raise FileNotFoundError("❌ CRITICAL: datacsv path not found in configuration")
    
    return True

# Default real data configuration
REAL_DATA_CONFIG = get_real_data_config()

# Export for easy import
__all__ = ['get_real_data_config', 'validate_real_data_config', 'REAL_DATA_CONFIG']
