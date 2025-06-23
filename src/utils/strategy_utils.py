"""
Utility functions for strategy logic (เทพ)
"""
import numpy as np
import pandas as pd

def safe_get_global(var_name, default_value):
    """Safely get a global variable, fallback to default if not found."""
    return globals().get(var_name, default_value)

# ...สามารถเพิ่ม utility อื่น ๆ ที่เกี่ยวข้องกับ strategy logic ได้ที่นี่...
