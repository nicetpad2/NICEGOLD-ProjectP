from typing import Any
import datetime
import json
import numpy as np
import pandas as pd
"""
Simple converter utility for JSON serialization
"""

def simple_converter(o: Any) -> Any:
    """
    Converts common pandas/numpy types for JSON serialization.

    Args:
        o: Object to convert

    Returns:
        JSON serializable version of the object
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        if np.isnan(o):
            return None
        if np.isinf(o):
            return "Infinity" if o > 0 else " - Infinity"
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, pd.Timedelta):
        return str(o)
    if isinstance(o, datetime.datetime):
        return o.isoformat()
    if isinstance(o, datetime.date):
        return o.isoformat()
    return o