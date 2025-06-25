# -*- coding: utf - 8 -* - 
# <<< เพิ่ม Encoding declaration สำหรับอักษรไทย (ควรอยู่บรรทัดแรกสุด) >>>
from dateutil.parser import parse as parse_date
from IPython import get_ipython
from projectp.utils_feature import map_standard_columns, assert_no_lowercase_columns
    import datetime
import datetime # <<< ENSURED Standard import 'import datetime'
import glob
import gzip
import json
import locale
import logging
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
    import requests
import subprocess
import sys
import traceback
"""Utility helpers for loading CSV files and preparing dataframes."""

# = =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
# = = = START OF PART 3/12 = =  = 
# = =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
# = = = PART 3: Helper Functions (Setup, Utils, Font, Config) (v4.8.8 - Patch 26.11 Applied) = =  = 
# = =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
# <<< MODIFIED v4.7.9: Implemented logging, added docstrings/comments, improved font setup robustness >>>
# <<< MODIFIED v4.7.9 (Post - Error): Corrected simple_converter for np.inf, then updated to 'Infinity' string with improved docstring >>>
# <<< MODIFIED v4.8.1: Added ValueError handling in parse_datetime_safely, refined simple_converter for NINF >>>
# <<< MODIFIED v4.8.2: Enhanced ValueError logging in parse_datetime_safely, refined simple_converter for NaN/Inf/NA/NaT and other types >>>
# <<< MODIFIED v4.8.3: Ensured simple_converter correctly handles np.inf/ - np.inf to string "Infinity"/" - Infinity" and other types for JSON.
#                      Corrected datetime import and usage. Re - indented and reviewed. Added Part Markers. >>>
# <<< MODIFIED v4.8.4: Added load_app_config function and updated versioning. >>>
# <<< MODIFIED v4.8.5: Added safe_get_global function definition. >>>
# <<< MODIFIED v4.8.8 (Patch 10): Refined safe_set_datetime to proactively handle column dtype to prevent FutureWarning. >>>
# <<< MODIFIED v4.8.8 (Patch 26.1): Corrected safe_set_datetime assignment to use pd.to_datetime(val).astype("datetime64[ns]") for robust dtype handling. >>>
# <<< MODIFIED v4.8.8 (Patch 26.3.1): Applied [PATCH A] to safe_set_datetime as per user prompt. >>>
# <<< MODIFIED v4.8.8 (Patch 26.4.1): Unified [PATCH A] for safe_set_datetime. >>>
# <<< MODIFIED v4.8.8 (Patch 26.5.1): Applied final [PATCH A] for safe_set_datetime from user prompt. >>>
# <<< MODIFIED v4.8.8 (Patch 26.7): Applied fix for FutureWarning in safe_set_datetime by ensuring column dtype is datetime64[ns] before assignment. >>>
# <<< MODIFIED v4.8.8 (Patch 26.8): Applied model_diagnostics_unit recommendation to safe_set_datetime for robust dtype handling. >>>
# <<< MODIFIED v4.8.8 (Patch 26.11): Further refined safe_set_datetime to more aggressively ensure column dtype is datetime64[ns] before assignment. >>>
try:
except ImportError:  # pragma: no cover - optional dependency for certain features
    requests = None

logger = logging.getLogger(__name__)

# - - - Locale Setup for Thai date parsing - -  - 
try:
    locale.setlocale(locale.LC_TIME, 'th_TH.UTF - 8')
except locale.Error:
    logging.debug("Locale th_TH not supported, falling back to default.")


# - - - Robust Thai date parser - -  - 
THAI_MONTH_MAP = {
    "ม.ค.": "01", 
    "ก.พ.": "02", 
    "มี.ค.": "03", 
    "เม.ย.": "04", 
    "พ.ค.": "05", 
    "มิ.ย.": "06", 
    "ก.ค.": "07", 
    "ส.ค.": "08", 
    "ก.ย.": "09", 
    "ต.ค.": "10", 
    "พ.ย.": "11", 
    "ธ.ค.": "12", 
}


def robust_date_parser(date_string):
    """Parse Thai date strings with ``dateutil``, handling Buddhist years."""
    normalized = str(date_string)
    for th, num in THAI_MONTH_MAP.items():
        if th in normalized:
            normalized = normalized.replace(th, num)
            break
    try:
        dt = parse_date(normalized, dayfirst = True)
    except Exception as e:
        raise ValueError(f"Cannot parse Thai date: {date_string}") from e
    if dt.year > 2500:
        dt = dt.replace(year = dt.year - 543)
    return dt

# - - - JSON Serialization Helper (moved earlier for global availability) - -  - 
# [Patch v5.2.2] Provide simple_converter for JSON dumps
def simple_converter(o):  # pragma: no cover
    """Converts common pandas/numpy types for JSON serialization."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        if np.isnan(o):
            return None
        if np.isinf(o):
            return "Infinity" if o > 0 else " - Infinity"
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, pd.Timedelta):
        return str(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if pd.isna(o):
        return None
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)

# - - - Helper for Safe Global Access (Defined *before* use in other parts) - -  - 
# <<< [Patch] ADDED v4.8.5: Function definition moved here >>>
def safe_get_global(var_name, default_value):
    """
    Safely retrieves a global variable from the current module's scope.

    Args:
        var_name (str): The name of the global variable to retrieve.
        default_value: The value to return if the global variable is not found.

    Returns:
        The value of the global variable or the default value.
    """
    try:
        # Use globals() which returns a dictionary representing the current global symbol table
        return globals().get(var_name, default_value)
    except Exception as e:
        # Log an error if there's an unexpected issue accessing globals()
        logging.error(f"   (Error) Unexpected error in safe_get_global for '{var_name}': {e}", exc_info = True)
        return default_value
# <<< End of [Patch] ADDED v4.8.5 >>>

# - - - Directory Setup Helper - -  - 
def setup_output_directory(base_dir, dir_name):
    """
    Creates the output directory if it doesn't exist and checks write permissions.

    Args:
        base_dir (str): The base directory path.
        dir_name (str): The name of the output directory to create within the base directory.

    Returns:
        str: The full path to the created/verified output directory.

    Raises:
        SystemExit: If the directory cannot be created or written to.
    """
    output_path = os.path.join(base_dir, dir_name)
    logging.info(f"   (Setup) กำลังตรวจสอบ/สร้าง Output Directory: {output_path}")
    try:
        os.makedirs(output_path, exist_ok = True)
        logging.info(f"      -> Directory exists or was created.")
        # Test write permissions
        test_file_path = os.path.join(output_path, ".write_test")
        with open(test_file_path, "w", encoding = "utf - 8") as f:
            f.write("test")
        # Remove the temporary file quietly regardless of location
        try:
            os.remove(test_file_path)
        except OSError:
            logging.debug(f"Unable to remove test file {test_file_path}")
        logging.info(f"      -> การเขียนไฟล์ทดสอบสำเร็จ.")
        return output_path
    except OSError as e:
        logging.error(f"   (Error) ไม่สามารถสร้างหรือเขียนใน Output Directory '{output_path}': {e}", exc_info = True)
        sys.exit(f"   ออก: ปัญหาการเข้าถึง Output Directory ({output_path}).")
    except Exception as e:
        logging.error(f"   (Error) เกิดข้อผิดพลาดที่ไม่คาดคิดระหว่างตั้งค่า Output Directory '{output_path}': {e}", exc_info = True)
        sys.exit(f"   ออก: ข้อผิดพลาดร้ายแรงในการตั้งค่า Output Directory ({output_path}).")

# - - - Font Setup Helpers - -  - 
# [Patch v5.0.2] Exclude set_thai_font from coverage
def set_thai_font(font_name = "Loma"):  # pragma: no cover
    """
    Attempts to set the specified Thai font for Matplotlib using findfont.
    Prioritizes specified font, then searches common fallbacks.

    Args:
        font_name (str): The preferred Thai font name. Defaults to "Loma".

    Returns:
        bool: True if a preferred or fallback font was successfully set and tested, False otherwise.
    """
    target_font_path = None
    actual_font_name = None
    # Added more common Thai fonts and ensured uniqueness
    preferred_fonts = [font_name] + ["TH Sarabun New", "THSarabunNew", "Garuda", "Norasi", "Kinnari", "Waree", "Laksaman", "Loma"]
    preferred_fonts = list(dict.fromkeys(preferred_fonts)) # Remove duplicates while preserving order
    logging.info(f"   [Font Check] Searching for preferred fonts: {preferred_fonts}")

    for pref_font in preferred_fonts:
        try:
            found_path = fm.findfont(pref_font, fallback_to_default = False)
            if found_path and os.path.exists(found_path):
                target_font_path = found_path
                prop = fm.FontProperties(fname = target_font_path)
                actual_font_name = prop.get_name()
                logging.info(f"      -> Found font: '{actual_font_name}' (requested: '{pref_font}') at path: {target_font_path}")
                break
        except ValueError:
            logging.debug(f"      -> Font '{pref_font}' not found by findfont.")
        except Exception as e_find: # Catch more general exceptions during findfont
            logging.warning(f"      -> Error finding font '{pref_font}': {e_find}")

    if target_font_path and actual_font_name:
        try:
            plt.rcParams['font.family'] = actual_font_name
            plt.rcParams['axes.unicode_minus'] = False # Important for correct display of minus sign
            logging.info(f"   Attempting to set default font to '{actual_font_name}'.")

            # Test plot to confirm font rendering
            fig_test, ax_test = plt.subplots(figsize = (0.5, 0.5)) # Small test figure
            ax_test.set_title(f"ทดสอบ ({actual_font_name})", fontname = actual_font_name)
            plt.close(fig_test) # Close the test figure immediately
            logging.info(f"      -> Font '{actual_font_name}' set and tested successfully.")
            return True
        except Exception as e_set:
            logging.warning(f"      -> (Warning) Font '{actual_font_name}' set, but test failed: {e_set}")
            # Attempt to revert to a known safe default if setting the Thai font fails
            try:
                plt.rcParams['font.family'] = 'DejaVu Sans' # A common fallback
                logging.info("         -> Reverted to 'DejaVu Sans' due to test failure.")
            except Exception as e_revert:
                logging.warning(f"         -> Failed to revert font to DejaVu Sans: {e_revert}")
            return False
    else:
        logging.warning(f"   (Warning) Could not find any suitable Thai fonts ({preferred_fonts}) using findfont.")
        return False

# [Patch v5.6.0] Split font installation and configuration helpers
def install_thai_fonts_colab():  # pragma: no cover
    """Install Thai fonts when running on Google Colab."""
    try:
        subprocess.run(["sudo", "apt - get", "update", " - qq"], check = False, capture_output = True, text = True, timeout = 120, encoding = 'utf - 8', errors = 'ignore')
        subprocess.run(["sudo", "apt - get", "install", " - y", " - qq", "fonts - thai - tlwg"], check = False, capture_output = True, text = True, timeout = 180, encoding = 'utf - 8', errors = 'ignore')
        subprocess.run(["fc - cache", " - fv"], check = False, capture_output = True, text = True, timeout = 120, encoding = 'utf - 8', errors = 'ignore')
        return True
    except Exception as e:
        logging.error(f"      (Error) Failed to install Thai fonts: {e}")
        return False


def configure_matplotlib_fonts(font_name = "TH Sarabun New"):  # pragma: no cover
    """Configure Matplotlib to use a given Thai font."""
    return set_thai_font(font_name)


def setup_fonts(output_dir = None):  # pragma: no cover
    """Sets up Thai fonts for Matplotlib plots."""
    logging.info("\n(Processing) Setting up Thai font for plots...")
    font_set_successfully = False
    preferred_font_name = "TH Sarabun New"
    try:
        ipython = get_ipython()
        in_colab = ipython is not None and 'google.colab' in str(ipython)
        font_set_successfully = configure_matplotlib_fonts(preferred_font_name)
        if not font_set_successfully and in_colab:
            logging.info("\n   Preferred font not found. Attempting installation via apt - get (Colab)...")
            if install_thai_fonts_colab():
                fm._load_fontmanager(try_read_cache = False)
                font_set_successfully = configure_matplotlib_fonts(preferred_font_name) or configure_matplotlib_fonts("Loma")
        if not font_set_successfully:
            fallback_fonts = ["Loma", "Garuda", "Norasi", "Kinnari", "Waree", "THSarabunNew"]
            logging.info(f"\n   Trying fallbacks ({', '.join(fallback_fonts)})...")
            for fb_font in fallback_fonts:
                if configure_matplotlib_fonts(fb_font):
                    font_set_successfully = True
                    break
        if not font_set_successfully:
            logging.critical("\n   (CRITICAL WARNING) Could not set any preferred Thai font. Plots WILL NOT render Thai characters correctly.")
        else:
            logging.info("\n   (Info) Font setup process complete.")
    except Exception as e:
        logging.error(f"   (Error) Critical error during font setup: {e}", exc_info = True)
# - - - ฟังก์ชันหลักถูกย้ายไปยัง src/data_loader/ - -  - 
# กรุณาแก้ไข import ในโปรเจกต์เป็น from src.data_loader.<module> import <function>
# For backward compatibility, you may import moved functions here if needed:
# from src.data_loader.csv_loader import safe_load_csv_auto, safe_load_csv, read_csv_with_date_parse, load_data_from_csv, read_csv_in_chunks
# from src.data_loader.project_loader import load_project_csvs, get_base_csv_data, load_raw_data_m1, load_raw_data_m15