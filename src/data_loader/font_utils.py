# Font utilities for data_loader module
import logging
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def set_thai_font(font_name="Loma"):  # pragma: no cover
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
    preferred_fonts = [font_name] + ["TH Sarabun New", "THSarabunNew", "Garuda", "Norasi", "Kinnari", "Waree", "Laksaman", "Loma"]
    preferred_fonts = list(dict.fromkeys(preferred_fonts))
    logging.info(f"   [Font Check] Searching for preferred fonts: {preferred_fonts}")
    for pref_font in preferred_fonts:
        try:
            found_path = fm.findfont(pref_font, fallback_to_default=False)
            if found_path and os.path.exists(found_path):
                target_font_path = found_path
                prop = fm.FontProperties(fname=target_font_path)
                actual_font_name = prop.get_name()
                logging.info(f"      -> Found font: '{actual_font_name}' (requested: '{pref_font}') at path: {target_font_path}")
                break
        except ValueError:
            logging.debug(f"      -> Font '{pref_font}' not found by findfont.")
        except Exception as e_find:
            logging.warning(f"      -> Error finding font '{pref_font}': {e_find}")
    if target_font_path and actual_font_name:
        try:
            plt.rcParams['font.family'] = actual_font_name
            plt.rcParams['axes.unicode_minus'] = False
            logging.info(f"   Attempting to set default font to '{actual_font_name}'.")
            fig_test, ax_test = plt.subplots(figsize=(0.5, 0.5))
            ax_test.set_title(f"ทดสอบ ({actual_font_name})", fontname=actual_font_name)
            plt.close(fig_test)
            logging.info(f"      -> Font '{actual_font_name}' set and tested successfully.")
            return True
        except Exception as e_set:
            logging.warning(f"      -> (Warning) Font '{actual_font_name}' set, but test failed: {e_set}")
            try:
                plt.rcParams['font.family'] = 'DejaVu Sans'
                logging.info("         -> Reverted to 'DejaVu Sans' due to test failure.")
            except Exception as e_revert:
                logging.warning(f"         -> Failed to revert font to DejaVu Sans: {e_revert}")
            return False
    else:
        logging.warning(f"   (Warning) Could not find any suitable Thai fonts ({preferred_fonts}) using findfont.")
        return False

def install_thai_fonts_colab():  # pragma: no cover
    """Install Thai fonts when running on Google Colab."""
    try:
        subprocess.run(["sudo", "apt-get", "update", "-qq"], check=False, capture_output=True, text=True, timeout=120)
        subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "fonts-thai-tlwg"], check=False, capture_output=True, text=True, timeout=180)
        subprocess.run(["fc-cache", "-fv"], check=False, capture_output=True, text=True, timeout=120)
        return True
    except Exception as e:
        logging.error(f"      (Error) Failed to install Thai fonts: {e}")
        return False

def configure_matplotlib_fonts(font_name="TH Sarabun New"):  # pragma: no cover
    """Configure Matplotlib to use a given Thai font."""
    return set_thai_font(font_name)

def setup_fonts(output_dir=None):  # pragma: no cover
    """Sets up Thai fonts for Matplotlib plots."""
    logging.info("\n(Processing) Setting up Thai font for plots...")
    font_set_successfully = False
    preferred_font_name = "TH Sarabun New"
    try:
        ipython = None
        try:
            from IPython import get_ipython
            ipython = get_ipython()
        except ImportError:
            pass
        in_colab = ipython is not None and 'google.colab' in str(ipython)
        font_set_successfully = configure_matplotlib_fonts(preferred_font_name)
        if not font_set_successfully and in_colab:
            logging.info("\n   Preferred font not found. Attempting installation via apt-get (Colab)...")
            if install_thai_fonts_colab():
                fm._load_fontmanager(try_read_cache=False)
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
        logging.error(f"   (Error) Critical error during font setup: {e}", exc_info=True)
